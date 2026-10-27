//! Progress of long runs (a clip through stage 1 and stage 2, …). Libraries only report
//! ([`ProgressSink`]); entry points draw it (a Python bar, a terminal bar) and estimate the time
//! left from the rate.

/// Where a run is: stage `step` of `steps` (1-based), named `stage`, with `done` of `total` units
/// of its work finished (`total` 0: unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressEvent<'a> {
    pub stage: &'a str,
    pub step: usize,
    pub steps: usize,
    pub done: u64,
    pub total: u64,
    /// What `done` counts (e.g. `"windows"`, `"frames"`).
    pub unit: &'a str,
}

/// Receives the progress of a run. Called from the run's thread; keep it short.
pub trait ProgressSink: Send + Sync {
    fn report(&self, event: &ProgressEvent<'_>);
}

/// Ignores progress.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn report(&self, _event: &ProgressEvent<'_>) {}
}

impl<F: Fn(&ProgressEvent<'_>) + Send + Sync> ProgressSink for F {
    fn report(&self, event: &ProgressEvent<'_>) {
        self(event)
    }
}

/// The stages of one run, numbered in order: reports through `sink` as
/// `stages.report(index, done, total)`.
#[derive(Clone, Copy)]
pub struct Stages<'a> {
    sink: &'a dyn ProgressSink,
    names: &'a [(&'a str, &'a str)],
}

impl<'a> Stages<'a> {
    /// `names`: `(stage, unit)` of every stage, in order.
    pub fn new(sink: &'a dyn ProgressSink, names: &'a [(&'a str, &'a str)]) -> Self {
        Self { sink, names }
    }

    /// Reports `done` of `total` for stage `name` (ignored when the run has no such stage).
    pub fn report(&self, name: &str, done: u64, total: u64) {
        if let Some(i) = self.names.iter().position(|(n, _)| *n == name) {
            let (stage, unit) = self.names[i];
            self.sink.report(&ProgressEvent { stage, step: i + 1, steps: self.names.len(), done, total, unit });
        }
    }
}
