//! Execution runtime and zero-copy tensor bridging between Burn (`CubeBackend`) and CubeCL
//! kernels.

pub mod bridge;

pub use bridge::{
    MlContext, elem_dtype, from_cube, into_contiguous_cube, into_cube,
};
