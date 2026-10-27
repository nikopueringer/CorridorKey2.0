//! What a clip is: its geometry ([`FrameMeta`]), its colour encoding ([`ColorSpace`],
//! [`LinearAdjust`]) and where its frames come from ([`FrameSource`]).

mod color;
mod meta;
mod source;

pub use color::{ColorSpace, LinearAdjust, HDR_ASINH_KNEE, HDR_ASINH_REF};
pub use meta::{FrameMeta, Padding, SHAPE_QUANTUM};
pub use source::FrameSource;
