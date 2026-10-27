use thiserror::Error;

pub type CkResult<T> = Result<T, CkError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CkError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Shape error: {0}")]
    Shape(String),

    #[error("Frame range {start}..{end} is outside 0..{total}")]
    FrameRange { start: usize, end: usize, total: usize },

    #[error("Device compute error: {0}")]
    ComputeError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Weights error: {0}")]
    Weights(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

impl From<std::io::Error> for CkError {
    fn from(e: std::io::Error) -> Self {
        CkError::Io(e.to_string())
    }
}

impl From<crate::device::ComputeError> for CkError {
    fn from(e: crate::device::ComputeError) -> Self {
        CkError::ComputeError(e.to_string())
    }
}
