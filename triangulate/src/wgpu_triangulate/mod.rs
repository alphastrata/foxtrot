pub mod cached_triangulation;
pub mod triangulation_utils;
pub mod wgpu_utils;

pub use triangulation_utils::{
    wgpu_triangulate, wgpu_triangulate_batch_from_examples, wgpu_triangulate_with_context,
};
pub use wgpu_utils::*;

// GPU compute pass to raise 2D points to 3D and apply transformations
pub fn triangulate(s: &step::step_file::StepFile) -> (crate::mesh::Mesh, crate::stats::Stats) {
    // Simply call the existing wgpu_triangulate function from the main module
    // This avoids reimplementing the complex logic while ensuring parity
    crate::wgpu_triangulate::wgpu_triangulate(s)
}
