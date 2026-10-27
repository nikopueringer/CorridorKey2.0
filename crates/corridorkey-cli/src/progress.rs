//! Terminal progress bar for long runs: one line per stage on stderr,
//! `[2/4] corridorkey_v2.safetensors  ████░░░░  1200/2660 MB  0:05 < 0:06`, the time left estimated from the
//! rate so far. Any run reporting through [`corridorkey_core::ProgressSink`] can draw it.

use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use corridorkey_core::{ProgressEvent, ProgressSink};

/// Characters of the bar.
const BAR_WIDTH: usize = 24;
/// Least time between two redraws of a stage's line.
const REDRAW: Duration = Duration::from_millis(100);
const SECS_PER_MINUTE: u64 = 60;
const SECS_PER_HOUR: u64 = 3600;

#[derive(Default)]
struct Line {
    /// `(step, stage)` drawn now.
    key: Option<(usize, String)>,
    start: Option<Instant>,
    drawn: Option<Instant>,
    width: usize,
}

/// Draws progress on stderr, one line per stage, ending each line when its stage completes.
#[derive(Default)]
pub struct TerminalProgress {
    line: Mutex<Line>,
}

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    let (h, m, s) = (s / SECS_PER_HOUR, s % SECS_PER_HOUR / SECS_PER_MINUTE, s % SECS_PER_MINUTE);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

impl ProgressSink for TerminalProgress {
    fn report(&self, e: &ProgressEvent<'_>) {
        let mut line = self.line.lock().expect("progress lock");
        let now = Instant::now();
        let key = (e.step, e.stage.to_string());
        if line.key.as_ref() != Some(&key) {
            if line.key.is_some() {
                eprintln!();
            }
            *line = Line { key: Some(key), start: Some(now), drawn: None, width: 0 };
        }
        let finished = e.total > 0 && e.done >= e.total;
        if !finished && line.drawn.is_some_and(|t| now - t < REDRAW) {
            return;
        }
        line.drawn = Some(now);
        let elapsed = now - line.start.unwrap_or(now);
        let label = format!("[{}/{}] {}", e.step, e.steps, e.stage);
        let text = if e.total > 0 {
            let filled = (BAR_WIDTH as u64 * e.done.min(e.total) / e.total) as usize;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled));
            let left = (e.done > 0 && e.done < e.total).then(|| format!(" < {}", clock(elapsed.mul_f64((e.total - e.done) as f64 / e.done as f64))));
            format!("{label}  {bar}  {}/{} {}  {}{}", e.done, e.total, e.unit, clock(elapsed), left.unwrap_or_default())
        } else {
            format!("{label}  {} {}  {}", e.done, e.unit, clock(elapsed))
        };
        let width = text.chars().count();
        // Pad over a longer previous line
        let mut err = std::io::stderr().lock();
        let _ = write!(err, "\r{text:<0$}", line.width.max(width));
        let _ = err.flush();
        line.width = width;
        if finished {
            let _ = writeln!(err);
            line.key = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_formats_minutes_and_hours() {
        assert_eq!(clock(Duration::from_secs(65)), "1:05");
        assert_eq!(clock(Duration::from_secs(3 * 3600 + 7)), "3:00:07");
    }
}
