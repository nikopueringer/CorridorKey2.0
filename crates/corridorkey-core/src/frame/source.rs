//! Where frames come from. Native builds decode with ffmpeg or read EXR folders; the browser
//! decodes with WebCodecs. Both supply frames through [`FrameSource`], so the pipeline does not
//! depend on either.

use super::{ColorSpace, FrameMeta};
use crate::CkResult;

/// A clip's decoded frames, in order.
pub trait FrameSource {
    /// Geometry of the clip.
    fn meta(&self) -> &FrameMeta;

    /// Encoding of the values [`Self::read`] returns.
    fn color_space(&self) -> ColorSpace;

    /// Writes frames `start .. start + count` into `out`, `[count, native_h, native_w, 3]` RGB,
    /// row-major (the decoders' order; the upload kernel converts it). Returns the frames read
    /// (fewer at the end of the clip).
    fn read(&mut self, start: usize, count: usize, out: &mut [f32]) -> CkResult<usize>;
}
