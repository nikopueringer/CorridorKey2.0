//! How tensors sit in device memory.
//!
//! Every activation has the logical shape `[batch, time, height, width, channels]`. The networks
//! keep one order end to end, [`MemoryOrder::TokensLast`]: linear layers, norms and attention all
//! read a token's channels contiguously, so no permute is needed between blocks. [`MemoryOrder::Planar`]
//! exists only at the edges (decoded frames, written outputs); converting between the two is an
//! explicit, named step, never a side effect of a reshape.
//!
//! [`SwinWindows`] maps window tokens to grid positions (cyclic shift and padding included) so
//! attention kernels read windows in place instead of copying them.

mod window;

pub use window::SwinWindows;

use serde::{Deserialize, Serialize};

use crate::{CkError, CkResult};

/// Element order of a `[batch, time, height, width, channels]` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOrder {
    /// `[B, T, H, W, C]`: channels of a token are contiguous. The order inside the networks.
    TokensLast,
    /// `[B, T, C, H, W]`: one plane per channel. Decoded frames and written outputs.
    Planar,
}

/// Shape, order and strides (in elements) of a 5-D tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorLayout {
    pub batch: usize,
    pub time: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub order: MemoryOrder,
}

impl TensorLayout {
    pub fn new(batch: usize, time: usize, height: usize, width: usize, channels: usize, order: MemoryOrder) -> Self {
        Self { batch, time, height, width, channels, order }
    }

    /// `[B, T, H, W, C]` in [`MemoryOrder::TokensLast`].
    pub fn tokens(batch: usize, time: usize, height: usize, width: usize, channels: usize) -> Self {
        Self::new(batch, time, height, width, channels, MemoryOrder::TokensLast)
    }

    /// Logical shape `[B, T, H, W, C]` (whatever the order).
    pub fn shape(&self) -> [usize; 5] {
        [self.batch, self.time, self.height, self.width, self.channels]
    }

    pub fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Tokens (`B·T·H·W`): the row count of the `[tokens, channels]` matrix a linear layer sees.
    pub fn num_tokens(&self) -> usize {
        self.batch * self.time * self.height * self.width
    }

    /// Strides of `[b, t, y, x, c]`, in elements.
    pub fn strides(&self) -> [usize; 5] {
        let (h, w, c) = (self.height, self.width, self.channels);
        match self.order {
            MemoryOrder::TokensLast => [self.time * h * w * c, h * w * c, w * c, c, 1],
            MemoryOrder::Planar => [self.time * c * h * w, c * h * w, w, 1, h * w],
        }
    }

    /// Linear index of element `[b, t, y, x, c]`.
    #[inline(always)]
    pub fn linear_index(&self, b: usize, t: usize, y: usize, x: usize, c: usize) -> usize {
        let s = self.strides();
        b * s[0] + t * s[1] + y * s[2] + x * s[3] + c * s[4]
    }

    /// The same tensor in `order`.
    pub fn with_order(self, order: MemoryOrder) -> Self {
        Self { order, ..self }
    }

    /// `Ok` if `other` has the same logical shape (the order may differ).
    pub fn check_shape(&self, other: &Self) -> CkResult<()> {
        if self.shape() == other.shape() {
            Ok(())
        } else {
            Err(CkError::ShapeMismatch { expected: self.shape().to_vec(), actual: other.shape().to_vec() })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every element has one index, and the indices cover the buffer exactly, in both orders.
    #[test]
    fn indices_are_a_bijection_in_both_orders() {
        for order in [MemoryOrder::TokensLast, MemoryOrder::Planar] {
            let l = TensorLayout::new(2, 3, 5, 7, 3, order);
            let mut seen = vec![false; l.num_elements()];
            for b in 0..2 {
                for t in 0..3 {
                    for y in 0..5 {
                        for x in 0..7 {
                            for c in 0..3 {
                                let i = l.linear_index(b, t, y, x, c);
                                assert!(!seen[i], "{order:?} index {i} repeated");
                                seen[i] = true;
                            }
                        }
                    }
                }
            }
            assert!(seen.into_iter().all(|s| s));
        }
        let tokens = TensorLayout::tokens(1, 1, 2, 2, 8);
        assert_eq!(tokens.linear_index(0, 0, 0, 1, 0), 8, "a token's channels are contiguous");
    }
}
