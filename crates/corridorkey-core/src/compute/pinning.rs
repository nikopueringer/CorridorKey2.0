//! Reproducible results: pinning the autotuned choices that change the numbers.
//!
//! Some kernels offer several equivalent implementations and let CubeCL's autotuner pick the
//! fastest per device. Most compute identical values, but a few sum in a different order (an IIR
//! filter's time blocks, a matrix product's routine), and the tuner can pick differently from run
//! to run. While a [`PinnedChoices`] guard lives, those kernels use fixed, documented choices on
//! the thread that holds it, so the same input on the same device gives the same output. Results
//! are not promised to match bit for bit across devices (different hardware rounds differently).

use std::cell::Cell;
use std::marker::PhantomData;

thread_local! {
    static PINNED: Cell<u32> = const { Cell::new(0) };
}

/// While alive, [`tuned_choices_pinned`] is `true` on this thread. Guards nest.
#[must_use = "the choices are pinned only while the guard lives"]
pub struct PinnedChoices {
    // Not `Send`: the setting belongs to the thread that launches the kernels
    _thread: PhantomData<*const ()>,
}

/// Pins result-changing autotuned choices on this thread until the guard is dropped.
pub fn pin_tuned_choices() -> PinnedChoices {
    PINNED.with(|p| p.set(p.get() + 1));
    PinnedChoices { _thread: PhantomData }
}

impl Drop for PinnedChoices {
    fn drop(&mut self) {
        PINNED.with(|p| p.set(p.get().saturating_sub(1)));
    }
}

/// Whether a [`PinnedChoices`] guard is alive on this thread.
pub fn tuned_choices_pinned() -> bool {
    PINNED.with(|p| p.get() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_nest_and_stay_on_their_thread() {
        assert!(!tuned_choices_pinned());
        let outer = pin_tuned_choices();
        {
            let _inner = pin_tuned_choices();
            assert!(tuned_choices_pinned());
            assert!(!std::thread::spawn(tuned_choices_pinned).join().unwrap(), "other threads are not pinned");
        }
        assert!(tuned_choices_pinned());
        drop(outer);
        assert!(!tuned_choices_pinned());
    }
}
