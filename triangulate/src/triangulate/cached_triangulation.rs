// Re-export for backward compatibility
// The actual implementation is in the wgpu_triangulate module
#[cfg(feature = "wgpu")]
pub use crate::wgpu_triangulate::cached_triangulation::*;
