//! `corridorkey_rs`: the CorridorKey Rust engine from Python.
//!
//! ```python
//! import corridorkey_rs as ck
//! ck.available_runtimes()                 # ["wgpu", "cuda"]
//! keyer = ck.Keyer(root=".", runtime="cuda")
//! keyer.key_clip("clip.mp4", "out", outputs=["alpha", "fg"])
//! mattes = keyer.key_frames(frames)       # uint8 (T, H, W, 3) -> {"alpha": float32 (T, H, W), ...}
//! ```
//!
//! Runtime rule for every call: an explicit `runtime=`, else the one chosen with `set_runtime`, else
//! `CORRIDORKEY_RUNTIME`, else the first compiled in (GPUs first). Keying releases the GIL.

use std::path::PathBuf;
use std::sync::Mutex;

use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{ComputeTarget, LinearAdjust};
use corridorkey_io::writers::EncoderConfig;
use corridorkey_io::{ClipKeyer, ClipOptions, MemoryHints, Precision, StartMode};
use corridorkey_ml::pipeline::ModelPaths;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The runtime chosen with `set_runtime`, if any.
static SELECTED: Mutex<Option<ComputeTarget>> = Mutex::new(None);

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn runtime_error(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn parse_runtime(name: &str) -> PyResult<ComputeTarget> {
    ComputeTarget::parse(name).and_then(ComputeTarget::checked).map_err(value_error)
}

/// The runtime a call uses: `explicit` if given, else the selected one, else the default.
fn target(explicit: Option<&str>) -> PyResult<ComputeTarget> {
    if let Some(name) = explicit {
        return parse_runtime(name);
    }
    if let Some(selected) = *SELECTED.lock().expect("runtime selection lock") {
        return Ok(selected);
    }
    ComputeTarget::from_env().map_err(value_error)
}

/// Names of the runtimes compiled into this build, in preference order (GPUs first).
#[pyfunction]
fn available_runtimes() -> Vec<&'static str> {
    ComputeTarget::available().into_iter().map(ComputeTarget::name).collect()
}

/// The runtime calls use when they are not given `runtime=`.
#[pyfunction]
fn current_runtime() -> PyResult<&'static str> {
    Ok(target(None)?.name())
}

/// Chooses the runtime of every later `Keyer` (unless given `runtime=`); `None` goes back to the
/// default (`CORRIDORKEY_RUNTIME`, else the first compiled in).
#[pyfunction]
#[pyo3(signature = (name))]
fn set_runtime(name: Option<&str>) -> PyResult<()> {
    let chosen = name.map(parse_runtime).transpose()?;
    *SELECTED.lock().expect("runtime selection lock") = chosen;
    Ok(())
}

fn parse_output(name: &str) -> PyResult<RenderOutput> {
    Ok(match name.to_lowercase().replace('-', "_").as_str() {
        "alpha" => RenderOutput::Alpha,
        "fg" | "raw_fg" | "foreground" => RenderOutput::Fg,
        "checker" => RenderOutput::Checker,
        "cutout" => RenderOutput::Cutout,
        other => return Err(value_error(format!("unknown output {other:?} (alpha, fg, checker, cutout)"))),
    })
}

fn output_name(output: RenderOutput) -> &'static str {
    match output {
        RenderOutput::Alpha => "alpha",
        RenderOutput::Fg => "fg",
        RenderOutput::Checker => "checker",
        RenderOutput::Cutout => "cutout",
    }
}

/// The CorridorKey networks loaded on one device, ready to key any number of clips.
///
/// Parameters
/// ----------
/// root : str
///     Package root holding `weights/` and `vfms/` (converted backbones).
/// runtime : str or None
///     `"cuda"`, `"wgpu"`, ... (see `available_runtimes()`); `None`: the current runtime.
/// precision : str
///     `"auto"` (f16 on GPUs that support it), `"f32"` or `"f16"`.
/// weights : str or None
///     Checkpoint, instead of `<root>/weights/corridorkey_v2.safetensors`.
#[pyclass(frozen, module = "corridorkey_rs")]
struct Keyer {
    inner: ClipKeyer,
}

#[pymethods]
impl Keyer {
    #[new]
    #[pyo3(signature = (root = ".".into(), runtime = None, precision = "auto", weights = None))]
    fn new(py: Python<'_>, root: PathBuf, runtime: Option<&str>, precision: &str, weights: Option<PathBuf>) -> PyResult<Self> {
        let target = target(runtime)?;
        let precision = match precision {
            "auto" => Precision::Auto,
            "f32" => Precision::F32,
            "f16" => Precision::F16,
            other => return Err(value_error(format!("unknown precision {other:?} (auto, f32, f16)"))),
        };
        let mut paths = ModelPaths::from_root(&root);
        if let Some(w) = weights {
            paths.checkpoint = w;
        }
        let inner = py.detach(|| ClipKeyer::load(target, &paths, precision)).map_err(runtime_error)?;
        Ok(Self { inner })
    }

    /// The runtime the networks run on.
    #[getter]
    fn runtime(&self) -> &'static str {
        self.inner.target().name()
    }

    /// `"f16"` or `"f32"`: what the networks compute in.
    #[getter]
    fn precision(&self) -> &'static str {
        match self.inner.precision() {
            Precision::F16 => "f16",
            _ => "f32",
        }
    }

    /// Keys a video file or a directory of PNG / JPEG / EXR frames into `output_dir`, with the same file
    /// names and encodings as the product's `infer.py`. Returns what was written:
    /// `{"clip", "frames", "width", "height", "fps", "written", "seconds"}`.
    #[pyo3(signature = (
        input, output_dir, *, outputs = vec!["alpha".to_string(), "fg".to_string()], num_frames = None,
        start_mode = "begin", seed = 0, frame_dir_fps = 24.0, frame_dir_linear = false, hann_chunk = None, hann_stride = None,
        hint_first_frame = None, hint_video = None, hint_quality = 0.95, carry_hint = true, carry_hint_quality = 0.95, stage2_batch = 1, low_vram = false,
        output_format = "video", exr_dtype = "half", despill_strength = 0.5, cutout_linear = false,
        linear_brightness = 1.0, linear_contrast = 1.0, linear_contrast_pivot = 0.18,
        ffmpeg = "ffmpeg".to_string(), ffmpeg_codec = "libx264".to_string(), crf = 12,
        preset = "medium".to_string(), pix_fmt = "yuv444p10le".to_string(), bitrate = None,
        ffmpeg_threads = 2, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn key_clip<'py>(
        &self,
        py: Python<'py>,
        input: PathBuf,
        output_dir: PathBuf,
        outputs: Vec<String>,
        num_frames: Option<usize>,
        start_mode: &str,
        seed: u64,
        frame_dir_fps: f64,
        frame_dir_linear: bool,
        hann_chunk: Option<usize>,
        hann_stride: Option<usize>,
        hint_first_frame: Option<PathBuf>,
        hint_video: Option<PathBuf>,
        hint_quality: f64,
        carry_hint: bool,
        carry_hint_quality: f64,
        stage2_batch: usize,
        low_vram: bool,
        output_format: &str,
        exr_dtype: &str,
        despill_strength: f64,
        cutout_linear: bool,
        linear_brightness: f64,
        linear_contrast: f64,
        linear_contrast_pivot: f64,
        ffmpeg: String,
        ffmpeg_codec: String,
        crf: u32,
        preset: String,
        pix_fmt: String,
        bitrate: Option<String>,
        ffmpeg_threads: u32,
        verbose: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let exr = match (output_format, exr_dtype) {
            ("video", _) => None,
            ("exr", "half") => Some(true),
            ("exr", "float") => Some(false),
            ("exr", other) => return Err(value_error(format!("unknown exr_dtype {other:?} (half, float)"))),
            (other, _) => return Err(value_error(format!("unknown output_format {other:?} (video, exr)"))),
        };
        let opts = ClipOptions {
            outputs: outputs.iter().map(|o| parse_output(o)).collect::<PyResult<_>>()?,
            num_frames,
            start_mode: match start_mode.replace('-', "_").as_str() {
                "begin" => StartMode::Begin,
                "middle" => StartMode::Middle,
                "random_middle" => StartMode::RandomMiddle,
                other => return Err(value_error(format!("unknown start_mode {other:?} (begin, middle, random_middle)"))),
            },
            seed,
            frame_dir_fps,
            frame_dir_linear,
            hann_chunk,
            hann_stride,
            hint_first_frame,
            hint_video,
            hint_quality,
            carry_hint,
            carry_hint_quality,
            stage2_batch,
            low_vram,
            exr,
            despill: despill_strength,
            cutout_linear,
            linear: LinearAdjust { gain: linear_brightness, contrast: linear_contrast, pivot: linear_contrast_pivot },
            encoder: EncoderConfig { ffmpeg, codec: ffmpeg_codec, crf, preset, pix_fmt, bitrate, threads: ffmpeg_threads },
            ..ClipOptions::new(input, output_dir)
        };
        let report = py
            .detach(|| {
                self.inner.key_clip(&opts, &mut |line| {
                    if verbose {
                        eprintln!("{line}");
                    }
                })
            })
            .map_err(runtime_error)?;
        let d = PyDict::new(py);
        d.set_item("clip", &report.clip)?;
        d.set_item("frames", report.frames)?;
        d.set_item("width", report.size.0)?;
        d.set_item("height", report.size.1)?;
        d.set_item("fps", report.fps)?;
        d.set_item("written", report.written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>())?;
        d.set_item("seconds", report.seconds)?;
        Ok(d)
    }

    /// Keys frames held in memory: `frames` is uint8 RGB of shape `(T, H, W, 3)` (display-referred,
    /// like decoded video). Optional float32 hints in [0, 1]: `hint_first_frame` `(H, W)` for frame
    /// 0, `hints` `(T, H, W)` for every frame, both trusted at `hint_quality`. Returns `{output: float32 array}`: `(T, H, W)` for alpha, `(T, H, W, 3)`
    /// RGB for fg / checker, `(T, H, W, 4)` RGBA for cutout, holding the values the EXR outputs hold
    /// (unclamped).
    #[pyo3(signature = (
        frames, *, outputs = vec!["alpha".to_string(), "fg".to_string()], hann_chunk = None,
        hann_stride = None, hint_first_frame = None, hints = None, hint_quality = 0.95, carry_hint = true, carry_hint_quality = 0.95,
        stage2_batch = 1, low_vram = false, despill_strength = 0.5, cutout_linear = false, linear_brightness = 1.0,
        linear_contrast = 1.0, linear_contrast_pivot = 0.18
    ))]
    #[allow(clippy::too_many_arguments)]
    fn key_frames<'py>(
        &self,
        py: Python<'py>,
        frames: PyReadonlyArray4<'py, u8>,
        outputs: Vec<String>,
        hann_chunk: Option<usize>,
        hann_stride: Option<usize>,
        hint_first_frame: Option<PyReadonlyArray2<'py, f32>>,
        hints: Option<PyReadonlyArray3<'py, f32>>,
        hint_quality: f64,
        carry_hint: bool,
        carry_hint_quality: f64,
        stage2_batch: usize,
        low_vram: bool,
        despill_strength: f64,
        cutout_linear: bool,
        linear_brightness: f64,
        linear_contrast: f64,
        linear_contrast_pivot: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = frames.shape().to_vec();
        let (t, h, w) = (shape[0], shape[1], shape[2]);
        if shape[3] != 3 {
            return Err(value_error(format!("frames must be (T, H, W, 3) RGB, got {shape:?}")));
        }
        let rgb = frames.as_slice().map_err(|_| value_error("frames must be C-contiguous (use numpy.ascontiguousarray)"))?;
        let opts = ClipOptions {
            outputs: outputs.iter().map(|o| parse_output(o)).collect::<PyResult<_>>()?,
            hann_chunk,
            hann_stride,
            hint_quality,
            carry_hint,
            carry_hint_quality,
            stage2_batch,
            low_vram,
            despill: despill_strength,
            cutout_linear,
            linear: LinearAdjust { gain: linear_brightness, contrast: linear_contrast, pivot: linear_contrast_pivot },
            ..ClipOptions::new("", "")
        };
        let contiguous = |name: &str| value_error(format!("{name} must be C-contiguous float32 (use numpy.ascontiguousarray)"));
        let first = hint_first_frame.as_ref().map(|a| a.as_slice().map_err(|_| contiguous("hint_first_frame"))).transpose()?;
        let every = hints.as_ref().map(|a| a.as_slice().map_err(|_| contiguous("hints"))).transpose()?;
        let memory_hints = MemoryHints { first_frame: first, frames: every };
        let mut keyed = py.detach(|| self.inner.key_frames(rgb, t, (h, w), memory_hints, &opts)).map_err(runtime_error)?;
        let d = PyDict::new(py);
        for o in &opts.outputs {
            let values = keyed.remove(o).unwrap_or_default();
            let c = o.channels(true);
            let array = values.into_pyarray(py);
            let array = if c == 1 { array.reshape([t, h, w])?.into_any() } else { array.reshape([t, h, w, c])?.into_any() };
            d.set_item(output_name(*o), array)?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("Keyer(runtime={:?}, precision={:?})", self.runtime(), self.precision())
    }
}

#[pymodule]
fn corridorkey_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(available_runtimes, m)?)?;
    m.add_function(wrap_pyfunction!(current_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(set_runtime, m)?)?;
    m.add_class::<Keyer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
