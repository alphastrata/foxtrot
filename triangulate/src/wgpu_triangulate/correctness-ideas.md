### Root Cause Analysis: The "Wheel" Artifact

The "wheel" or "starburst" artifact you're seeing is a classic symptom of a **failed geometric projection**. Your process involves taking 3D geometry, "lowering" it to a 2D representation for triangulation, and then "raising" it back to 3D. The wheel appears when all the vertices of a face are incorrectly lowered to the same point in 2D space (usually the origin, `(0,0)`).

When the 2D triangulation algorithm receives a set of boundary points and a collapsed center, it does the only thing it can: connect every boundary point to that central point, creating the characteristic radial pattern.

Here’s the likely chain of failure in your `wgpu` implementation:

1.  **Missing `prepare()` Step:** In the CPU-only code (`surface.rs`), before `lower_verts` is called, there's a crucial function: `prepare()`. This function inspects the actual vertices of a face to compute the correct transformation matrices (`mat` and `mat_i`) for complex surfaces like spheres and tori. These matrices establish a local coordinate system for the face.
2.  **Incorrect `GpuSurface` Conversion:** Your `wgpu_impl.rs::to_gpu_surface` function is a "stateless" conversion. It takes a `Surface` and converts it without the context of the vertices. For spheres and tori, this means it's sending an uninitialized or incorrect identity matrix to the GPU.
3.  **Faulty GPU Lowering:** When the `surface_lowering.wgsl` shader receives this incorrect `GpuSurface` uniform, its calculations (especially for `SURFACE_TYPE_SPHERE`) fail, projecting every 3D vertex to the UV origin `(0,0)`.
4.  **Redundant and Flawed Workflow:** The overall workflow in `triangulate_faces` is the core architectural problem. It correctly triangulates the face on the CPU, but then it takes the resulting correct 3D vertices and sends them *back* to the GPU to be lowered *again* using the faulty lowering shader. This corrupts the perfectly good geometry. The GPU then tries to raise these corrupted `(0,0)` UVs, resulting in the wheel.

**In short: The GPU is re-doing work the CPU has already done correctly, but the GPU is doing it with incorrect information.**

---

### Corrective Action Plan

The solution is to simplify the GPU's role. The CPU is excellent at the complex, conditional logic of triangulation. The GPU is excellent at simple, massively parallel math. We will leverage both for their strengths.

1.  **CPU's Role:** Generate a single, correct "template" mesh for each unique face, complete with accurate 3D vertex positions and normals.
2.  **GPU's Role:** Take the template mesh and a list of transformations and apply them in parallel to create all the final instances of that mesh. The GPU should **not** be involved in lowering or raising.

Here is the step-by-step implementation plan.

#### Step 1: Create a Simplified Transformation Shader

The existing `raising` and `lowering` shaders are overly complex for what we need. Let's create a single, focused shader whose only job is to apply transformations.

**Create a new file: `./triangulate/src/wgpu_triangulate/transform_mesh.wgsl`**
```wgsl
// A vertex from the CPU-generated template mesh
struct GpuTemplateVertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
};

// A transform to apply
struct GpuTransform {
    // The main transformation matrix for positions
    transform: mat4x4<f32>,
    // The inverse-transpose of the transform, for correctly transforming normals
    inverse_transpose: mat4x4<f32>,
};

// The final output vertex
struct GpuOutputVertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
};

// --- Buffers ---
@group(0) @binding(0) var<storage, read> template_verts: array<GpuTemplateVertex>;
@group(0) @binding(1) var<storage, read> transforms: array<GpuTransform>;
@group(0) @binding(2) var<storage, read_write> output_verts: array<GpuOutputVertex>;


@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;

    let num_template_verts = arrayLength(&template_verts);
    if (num_template_verts == 0u) {
        return;
    }

    // Determine which instance and template vertex this invocation corresponds to
    let instance_idx = out_idx / num_template_verts;
    let template_v_idx = out_idx % num_template_verts;

    if (instance_idx >= arrayLength(&transforms)) {
        return;
    }

    // Fetch the data
    let xform = transforms[instance_idx];
    let template_v = template_verts[template_v_idx];

    // Apply transformations
    let world_pos = xform.transform * template_v.pos;
    let world_norm = normalize(xform.inverse_transpose * template_v.norm);

    // Write the result
    output_verts[out_idx].pos = world_pos;
    output_verts[out_idx].norm = world_norm;
}```

#### Step 2: Refactor the `wgpu_impl.rs` Orchestration Logic

Now, we'll gut the existing `wgpu_impl.rs` file, removing the faulty lowering/raising functions and replacing them with a single `gpu_transform_mesh` function that uses our new shader.

**File: `./triangulate/src/wgpu_triangulate/wgpu_impl.rs`** (heavily modified)
```rust
use crate::mesh::{Mesh, Triangle, Vertex};
use crate::stats::Stats;
use crate::triangulate::advanced_face_to_mesh as cpu_advanced_face_to_mesh;
use nalgebra_glm as glm;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use step::step_file::StepFile;
use wgpu::util::DeviceExt;

// --- New GPU Data Structures ---

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTransform {
    transform: [[f32; 4]; 4],
    inverse_transpose: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    pos: [f32; 4],
    norm: [f32; 4],
}

// --- Helper Functions (Keep these) ---

pub fn create_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    // ... implementation is correct
}

pub fn mat_to_f32_array(m: &glm::DMat4) -> [[f32; 4]; 4] {
    // ... implementation is correct
}

// --- New GPU Transformation Function ---

async fn gpu_transform_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    template_mesh: &Mesh,
    transforms: &[glm::DMat4],
    color: glm::DVec3,
) -> Result<Vec<Vertex>, Box<dyn std::error::Error>> {
    if template_mesh.verts.is_empty() || transforms.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Prepare data for the GPU
    let gpu_template_verts: Vec<GpuVertex> = template_mesh
        .verts
        .iter()
        .map(|v| GpuVertex {
            pos: [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32, 1.0],
            norm: [v.norm.x as f32, v.norm.y as f32, v.norm.z as f32, 0.0],
        })
        .collect();

    let gpu_transforms: Vec<GpuTransform> = transforms
        .iter()
        .map(|t| {
            let inv_transpose = t.try_inverse().map(|inv| inv.transpose()).unwrap_or_else(glm::DMat4::identity);
            GpuTransform {
                transform: mat_to_f32_array(t),
                inverse_transpose: mat_to_f32_array(&inv_transpose),
            }
        })
        .collect();

    // 2. Create GPU buffers
    let template_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Template Vertex Buffer"),
        contents: bytemuck::cast_slice(&gpu_template_verts),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Transform Buffer"),
        contents: bytemuck::cast_slice(&gpu_transforms),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let num_output_verts = template_mesh.verts.len() * transforms.len();
    let output_buffer_size = (std::mem::size_of::<GpuVertex>() * num_output_verts) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Vertex Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // 3. Setup wgpu pipeline
    let shader_code = std::include_str!("transform_mesh.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Transform Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });
    
    // ... The rest of this function involves creating the bind group layout,
    // bind group, pipeline, command encoder, dispatching the compute job,
    // copying the result to a staging buffer, and mapping it back to the CPU.
    // This logic is lengthy but standard wgpu boilerplate. It can be adapted
    // directly from your original `gpu_raise_and_transform` function, just
    // with the new buffers and shader.

    // 4. After reading data back from staging buffer:
    // let result_data: Vec<GpuVertex> = ...
    // let final_vertices = result_data.iter().map(|gpu_v| Vertex {
    //     pos: glm::DVec3::new(gpu_v.pos[0] as f64, gpu_v.pos[1] as f64, gpu_v.pos[2] as f64),
    //     norm: glm::DVec3::new(gpu_v.norm[0] as f64, gpu_v.norm[1] as f64, gpu_v.norm[2] as f64),
    //     color,
    // }).collect();
    // Ok(final_vertices)

    // For now, let's placeholder the full wgpu boilerplate
    // to focus on the core logic.
    panic!("Full wgpu boilerplate for dispatch and data retrieval needs to be implemented here.");
}

// --- Main Triangulation Function (Corrected) ---

pub fn triangulate_faces(
    s: &StepFile,
    face_tasks: &Vec<crate::triangulate::FaceTask>,
) -> (Mesh, Stats) {
    let (device, queue) = create_wgpu_device().expect("Failed to create wgpu device");

    let total_errors = AtomicUsize::new(0);

    let meshes: Vec<Mesh> = face_tasks
        .par_iter()
        .filter_map(|task| {
            let mut template_mesh = Mesh::default();
            let mut stats = Stats::default();

            // STEP 1: Use the reliable CPU implementation to generate a correct "template" mesh.
            if cpu_advanced_face_to_mesh(s, task.face_id, &mut template_mesh.verts, &mut template_mesh.triangles, &mut stats).is_err() {
                total_errors.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            if template_mesh.verts.is_empty() {
                return Some(Mesh::default());
            }

            // Flip normals on the template if necessary.
            if task.flip_normal {
                for v in &mut template_mesh.verts {
                    v.norm = -v.norm;
                }
            }
            
            // STEP 2: Hand off the simple, parallelizable work to the GPU.
            let transformed_verts = match pollster::block_on(gpu_transform_mesh(
                &device,
                &queue,
                &template_mesh,
                &task.transforms,
                task.color,
            )) {
                Ok(verts) => verts,
                Err(e) => {
                    log::error!("GPU transform for face failed: {}", e);
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            let mut final_mesh = Mesh {
                verts: transformed_verts,
                triangles: Vec::with_capacity(template_mesh.triangles.len() * task.transforms.len()),
            };

            // STEP 3: Replicate triangle indices on the CPU. This is extremely fast.
            let num_template_verts = template_mesh.verts.len() as u32;
            for i in 0..task.transforms.len() {
                let v_offset = i as u32 * num_template_verts;
                for t in &template_mesh.triangles {
                    final_mesh.triangles.push(Triangle {
                        verts: t.verts.add_scalar(v_offset),
                    });
                }
            }
            
            Some(final_mesh)
        })
        .collect();

    // STEP 4: Combine the results from all tasks.
    let final_mesh = meshes.into_par_iter().reduce(Mesh::default, Mesh::combine);

    let stats = Stats {
        num_faces: face_tasks.len(),
        num_errors: total_errors.load(Ordering::Relaxed),
        ..Default::default()
    };

    (final_mesh, stats)
}
```
