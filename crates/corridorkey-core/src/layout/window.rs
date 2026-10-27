//! Swin window indexing without copies.
//!
//! The Python model (`blocks.py`) rolls the grid by the shift, pads it to a multiple of the window,
//! permutes it into `[windows, window², C]` and builds region ids the same way to mask attention
//! across the wrap-around. Each of those is a full copy of the activations per block. Here the
//! same mapping is index arithmetic: a kernel asks which grid position window token `(window,
//! local)` reads, and which region it belongs to, and reads the activations where they are.

/// Window partition of an `height × width` token grid into `window × window` windows, after a
/// cyclic shift of `(shift_h, shift_w)` and padding up to a multiple of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwinWindows {
    /// Real grid size (tokens).
    pub height: usize,
    pub width: usize,
    /// Window side (tokens).
    pub window: usize,
    /// Cyclic shift; `0` for unshifted blocks, usually `window / 2` for shifted ones.
    pub shift_h: usize,
    pub shift_w: usize,
}

impl SwinWindows {
    /// # Panics
    /// If `window` is 0 or a shift is not smaller than the window.
    pub fn new(height: usize, width: usize, window: usize, shift_h: usize, shift_w: usize) -> Self {
        assert!(window > 0, "window must be positive");
        assert!(shift_h < window && shift_w < window, "shift ({shift_h}, {shift_w}) must be smaller than the window {window}");
        Self { height, width, window, shift_h, shift_w }
    }

    /// Grid size padded up to a multiple of the window.
    pub fn padded(&self) -> (usize, usize) {
        (self.height.div_ceil(self.window) * self.window, self.width.div_ceil(self.window) * self.window)
    }

    /// Windows along `(y, x)`.
    pub fn windows_hw(&self) -> (usize, usize) {
        let (hp, wp) = self.padded();
        (hp / self.window, wp / self.window)
    }

    pub fn num_windows(&self) -> usize {
        let (nh, nw) = self.windows_hw();
        nh * nw
    }

    /// Tokens per window (`window²`).
    pub fn window_tokens(&self) -> usize {
        self.window * self.window
    }

    /// Padded-grid position `(y, x)` read by token `local` (row-major in its window) of window
    /// `index` (row-major over windows): the shift is undone with wrap-around on the padded grid.
    #[inline]
    pub fn padded_position(&self, index: usize, local: usize) -> (usize, usize) {
        let (_, nw) = self.windows_hw();
        let (hp, wp) = self.padded();
        let y = (index / nw) * self.window + local / self.window;
        let x = (index % nw) * self.window + local % self.window;
        ((y + self.shift_h) % hp, (x + self.shift_w) % wp)
    }

    /// Real grid position read by window token `(index, local)`, or `None` for padding.
    #[inline]
    pub fn position(&self, index: usize, local: usize) -> Option<(usize, usize)> {
        let (y, x) = self.padded_position(index, local);
        (y < self.height && x < self.width).then_some((y, x))
    }

    /// Region id of window token `(index, local)` for shifted-window masking: tokens of one window
    /// attend to each other only when their ids are equal and non-negative; `-1` is padding.
    /// Unshifted windows have one region (`0`). Matches `_shifted_region_ids_2d` in `blocks.py`.
    #[inline]
    pub fn region_id(&self, index: usize, local: usize) -> i32 {
        let (y, x) = self.padded_position(index, local);
        if y >= self.height || x >= self.width {
            return -1;
        }
        let (hp, wp) = self.padded();
        let band = |p: usize, size: usize, shift: usize| -> usize {
            if shift == 0 || p < size - self.window {
                0
            } else if p < size - shift {
                1
            } else {
                2
            }
        };
        let bands_w = if self.shift_w > 0 { 3 } else { 1 };
        (band(y, hp, self.shift_h) * bands_w + band(x, wp, self.shift_w)) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the Python steps done literally (pad, label, roll by −shift, partition).
    fn reference(h: usize, w: usize, win: usize, sh: usize, sw: usize) -> Vec<Vec<(i64, i32)>> {
        let (hp, wp) = (h.div_ceil(win) * win, w.div_ceil(win) * win);
        let slices = |size: usize, s: usize| -> Vec<std::ops::Range<usize>> {
            if s > 0 { vec![0..size - win, size - win..size - s, size - s..size] } else { vec![0..size] }
        };
        let mut region = vec![vec![-1i32; wp]; hp];
        let mut idx = 0;
        for hs in slices(hp, sh) {
            for ws in slices(wp, sw) {
                for y in hs.clone() {
                    for x in ws.clone() {
                        if y < h && x < w {
                            region[y][x] = idx;
                        }
                    }
                }
                idx += 1;
            }
        }
        // Source position labels: y·wp + x, −1 for padding
        let label = |y: usize, x: usize| if y < h && x < w { (y * wp + x) as i64 } else { -1 };
        // roll by (−sh, −sw): out[y][x] = in[(y + sh) % hp][(x + sw) % wp]
        let mut windows = vec![];
        for wy in 0..hp / win {
            for wx in 0..wp / win {
                let mut tokens = vec![];
                for ly in 0..win {
                    for lx in 0..win {
                        let (y, x) = ((wy * win + ly + sh) % hp, (wx * win + lx + sw) % wp);
                        let id = if sh == 0 && sw == 0 { if y < h && x < w { 0 } else { -1 } } else { region[y][x] };
                        tokens.push((label(y, x), id));
                    }
                }
                windows.push(tokens);
            }
        }
        windows
    }

    #[test]
    fn matches_python_partition_and_region_ids() {
        // Odd sizes, padding, shifts on one or both axes
        for &(h, w, win, sh, sw) in &[(8, 8, 4, 0, 0), (8, 8, 4, 2, 2), (9, 13, 4, 2, 1), (5, 7, 4, 0, 2), (16, 16, 16, 8, 8)] {
            let s = SwinWindows::new(h, w, win, sh, sw);
            let (_, wp) = s.padded();
            let reference = reference(h, w, win, sh, sw);
            assert_eq!(reference.len(), s.num_windows());
            for (index, tokens) in reference.iter().enumerate() {
                for (local, &(label, id)) in tokens.iter().enumerate() {
                    let got = s.position(index, local).map_or(-1, |(y, x)| (y * wp + x) as i64);
                    assert_eq!(got, label, "position {h}x{w} w{win} s({sh},{sw}) window {index} token {local}");
                    assert_eq!(s.region_id(index, local), id, "region {h}x{w} w{win} s({sh},{sw}) window {index} token {local}");
                }
            }
        }
    }
}
