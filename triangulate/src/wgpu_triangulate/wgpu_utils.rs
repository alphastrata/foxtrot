use log::trace;
use nalgebra_glm as glm;
use std::sync::atomic::{AtomicUsize, Ordering};
use wgpu::PollType;
use wgpu::util::DeviceExt;

use crate::mesh::{Mesh, Vertex};
use crate::stats::Stats;
use crate::surface::Surface;
use crate::triangulate::{
    advanced_face_to_mesh as cpu_advanced_face_to_mesh, get_surface as cpu_get_surface,
};
use step::{ap214::*, step_file::StepFile};

// Surface type constants for shader
const SURFACE_TYPE_PLANE: u32 = 0;
const SURFACE_TYPE_CYLINDER: u32 = 1;
const SURFACE_TYPE_SPHERE: u32 = 2;
const SURFACE_TYPE_TORUS: u32 = 3;
const SURFACE_TYPE_CONE: u32 = 4;
const SURFACE_TYPE_BSPLINE: u32 = 5;
const SURFACE_TYPE_NURBS: u32 = 6;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSurface {
    surface_type: u32,
    radius: f32,
    major_radius: f32,
    minor_radius: f32,
    mat: [[f32; 4]; 4],   // world from uv
    mat_i: [[f32; 4]; 4], // uv from world
    z_min: f32,
    z_max: f32,
    angle: f32,
    _p0: u32,           // padding
    location: [f32; 3], // additional location data
    _p1: f32,           // padding
    axis: [f32; 3],     // additional axis data
    _p2: f32,           // padding
}

// Structure for input vertex data
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTemplateVertex {
    pos: [f32; 4],   // 3D position (xyz) + padding
    norm: [f32; 4],  // normal (xyz) + padding
    color: [f32; 4], // color (rgb) + padding
}

// Structure for output (transformed vertices)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuOutputVertex {
    pos: [f32; 4],      // 3D position (xyz) + padding
    norm: [f32; 4],     // normal (xyz) + padding
    color: [f32; 4],    // color (rgb) + padding
    _padding: [f32; 4], // padding
}

// Helper to convert from `nalgebra_glm::DMat4` to `[[f32; 4]; 4]`
pub fn mat_to_f32_array(m: &glm::DMat4) -> [[f32; 4]; 4] {
    let m_f64 = m.as_slice(); // Get the slice of f64 values
    [
        [
            m_f64[0] as f32,
            m_f64[1] as f32,
            m_f64[2] as f32,
            m_f64[3] as f32,
        ],
        [
            m_f64[4] as f32,
            m_f64[5] as f32,
            m_f64[6] as f32,
            m_f64[7] as f32,
        ],
        [
            m_f64[8] as f32,
            m_f64[9] as f32,
            m_f64[10] as f32,
            m_f64[11] as f32,
        ],
        [
            m_f64[12] as f32,
            m_f64[13] as f32,
            m_f64[14] as f32,
            m_f64[15] as f32,
        ],
    ]
}

// Helper to convert `DVec3` to `[f32; 3]`
pub fn vec3_to_f32_array(v: &glm::DVec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

// Add a function to convert `surface::Surface` to `GpuSurface`
pub fn to_gpu_surface(surf: &Surface) -> GpuSurface {
    match surf {
        Surface::Plane { normal, mat_i } => {
            let mat = glm::inverse(mat_i);
            GpuSurface {
                surface_type: SURFACE_TYPE_PLANE,
                mat: mat_to_f32_array(&mat),
                mat_i: mat_to_f32_array(mat_i),
                axis: vec3_to_f32_array(normal),
                radius: 0.0,
                major_radius: 0.0,
                minor_radius: 0.0,
                z_min: 0.0,
                z_max: 0.0,
                angle: 0.0,
                location: [0.0, 0.0, 0.0], // will be set if needed
                _p0: 0,
                _p1: 0.0,
                _p2: 0.0,
            }
        }
        Surface::Cylinder {
            location,
            axis,
            mat,
            mat_i,
            radius,
            z_min,
            z_max,
        } => GpuSurface {
            surface_type: SURFACE_TYPE_CYLINDER,
            location: vec3_to_f32_array(location),
            axis: vec3_to_f32_array(axis),
            mat: mat_to_f32_array(mat),
            mat_i: mat_to_f32_array(mat_i),
            radius: *radius as f32,
            z_min: *z_min as f32,
            z_max: *z_max as f32,
            major_radius: 0.0,
            minor_radius: 0.0,
            angle: 0.0,
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
        Surface::Cone { mat, mat_i, angle } => GpuSurface {
            surface_type: SURFACE_TYPE_CONE,
            mat: mat_to_f32_array(mat),
            mat_i: mat_to_f32_array(mat_i),
            angle: *angle as f32,
            radius: 0.0,
            major_radius: 0.0,
            minor_radius: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            location: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 0.0],
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
        Surface::Sphere {
            location,
            mat,
            mat_i,
            radius,
        } => GpuSurface {
            surface_type: SURFACE_TYPE_SPHERE,
            location: vec3_to_f32_array(location),
            mat: mat_to_f32_array(mat),
            mat_i: mat_to_f32_array(mat_i),
            radius: *radius as f32,
            major_radius: 0.0,
            minor_radius: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            angle: 0.0,
            axis: [0.0, 0.0, 0.0],
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
        Surface::Torus {
            axis,
            location,
            mat,
            mat_i,
            major_radius,
            minor_radius,
        } => GpuSurface {
            surface_type: SURFACE_TYPE_TORUS,
            axis: vec3_to_f32_array(axis),
            location: vec3_to_f32_array(location),
            mat: mat_to_f32_array(mat),
            mat_i: mat_to_f32_array(mat_i),
            major_radius: *major_radius as f32,
            minor_radius: *minor_radius as f32,
            radius: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            angle: 0.0,
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
        Surface::BSpline(_) => GpuSurface {
            surface_type: SURFACE_TYPE_BSPLINE,
            radius: 0.0,
            major_radius: 0.0,
            minor_radius: 0.0,
            mat: [[0.0; 4]; 4],
            mat_i: [[0.0; 4]; 4],
            z_min: 0.0,
            z_max: 0.0,
            angle: 0.0,
            location: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 0.0],
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
        Surface::NURBS(_) => GpuSurface {
            surface_type: SURFACE_TYPE_NURBS,
            radius: 0.0,
            major_radius: 0.0,
            minor_radius: 0.0,
            mat: [[0.0; 4]; 4],
            mat_i: [[0.0; 4]; 4],
            z_min: 0.0,
            z_max: 0.0,
            angle: 0.0,
            location: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 0.0],
            _p0: 0,   // padding
            _p1: 0.0, // padding
            _p2: 0.0, // padding
        },
    }
}

pub fn create_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .expect("Failed to create device");

    Ok((device, queue))
}

// Structure for GPU transforms
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTransform {
    transform: [[f32; 4]; 4],
    inverse_transpose: [[f32; 4]; 4], // inverse transpose for normal transformation
}

// GPU Context manager for reusing GPU resources across multiple triangulation calls
pub struct GPUContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GPUContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (device, queue) = create_wgpu_device()?;
        Ok(GPUContext { device, queue })
    }

    pub fn triangulate(&self, step_file: &StepFile) -> (Mesh, Stats) {
        // Simply call the existing batch function with the shared GPU context
        // Extract face_tasks using the same logic as wgpu_triangulate
        use crate::triangulate::FaceTask;
        use crate::triangulate::{
            collect_faces_from_brep, presentation_style_color, transform_stack_roots,
        };
        use ahash::AHashMap;
        use step::step_file::FromEntity;

        // Build face catalog with minimal allocations (same as original)
        let brep_colors: AHashMap<_, nalgebra_glm::DVec3> = step_file
            .0
            .iter()
            .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
            .flat_map(|m| m.items.iter())
            .filter_map(|item| step_file.entity(item.cast::<StyledItem_>()))
            .filter_map(|styled| {
                if styled.styles.len() == 1 {
                    presentation_style_color(step_file, styled.styles[0]).map(|c| (styled.item, c))
                } else {
                    None
                }
            })
            .collect();

        let mut transform_stack = crate::triangulate::build_transform_stack(step_file, false);
        let mut roots = transform_stack_roots(&transform_stack);
        if roots.len() > 1 {
            transform_stack = crate::triangulate::build_transform_stack(step_file, true);
            roots = transform_stack_roots(&transform_stack);
        }

        let mut todo: Vec<_> = roots
            .into_iter()
            .map(|v| (v, nalgebra_glm::DMat4::identity()))
            .collect();
        let mut shape_rep_relationship: AHashMap<step::id::Id<_>, Vec<step::id::Id<_>>> =
            AHashMap::new();
        for (r1, r2) in step_file
            .0
            .iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
        {
            shape_rep_relationship.entry(r1).or_default().push(r2);
        }

        let mut to_mesh: AHashMap<step::id::Id<_>, Vec<nalgebra_glm::DMat4>> = AHashMap::new();
        while let Some((id, mat)) = todo.pop() {
            for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
                todo.push((*child, mat));
            }
            if let Some(children) = transform_stack.get(&id) {
                for (child, next_mat) in children {
                    todo.push((*child, mat * next_mat));
                }
            } else {
                let items = match &step_file[id] {
                    Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                    Entity::ShapeRepresentation(b) => &b.items,
                    Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                    _ => continue,
                };

                for m in items.iter() {
                    if matches!(
                        &step_file[*m],
                        Entity::ManifoldSolidBrep(_)
                            | Entity::BrepWithVoids(_)
                            | Entity::ShellBasedSurfaceModel(_)
                    ) {
                        to_mesh.entry(*m).or_default().push(mat);
                    }
                }
            }
        }

        if to_mesh.is_empty() {
            to_mesh = step_file
                .0
                .iter()
                .enumerate()
                .filter(|(_i, e)| {
                    matches!(
                        e,
                        Entity::ManifoldSolidBrep(_)
                            | Entity::BrepWithVoids(_)
                            | Entity::ShellBasedSurfaceModel(_)
                    )
                })
                .map(|(i, _e)| (step::id::Id::new(i), vec![nalgebra_glm::DMat4::identity()]))
                .collect();
        }

        // Extract all face IDs with metadata (no deep copies)
        let face_tasks: Vec<FaceTask> = to_mesh
            .into_iter()
            .flat_map(|(brep_id, mats)| {
                let color = brep_colors
                    .get(&brep_id)
                    .copied()
                    .unwrap_or(nalgebra_glm::DVec3::new(0.5, 0.5, 0.5));

                collect_faces_from_brep(step_file, brep_id).into_iter().map(
                    move |(face_id, flip)| FaceTask {
                        face_id,
                        transforms: mats.clone(),
                        color,
                        flip_normal: flip,
                    },
                )
            })
            .collect();

        // Use the existing function with the shared GPU device
        triangulate_faces_with_device(step_file, &face_tasks, &self.device, &self.queue)
    }

    pub fn triangulate_faces(
        &self,
        step_file: &StepFile,
        face_tasks: &Vec<crate::triangulate::FaceTask>,
    ) -> (Mesh, Stats) {
        triangulate_faces_with_device(step_file, face_tasks, &self.device, &self.queue)
    }
}

pub fn triangulate_faces(
    s: &StepFile,
    face_tasks: &Vec<crate::triangulate::FaceTask>,
) -> (Mesh, Stats) {
    // Initialize wgpu
    let (device, queue) = create_wgpu_device().expect("Failed to create wgpu device");

    // Use the function with the shared GPU device
    triangulate_faces_with_device(s, face_tasks, &device, &queue)
}

pub async fn gpu_transform_vertices(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertices: &[Vertex],
    _surface: &Surface, // Not used in this corrected implementation
    transforms: &[glm::DMat4],
) -> Result<Vec<Vertex>, Box<dyn std::error::Error>> {
    trace!("gpu_transform_vertices: Starting");

    // Convert input vertices to GPU format
    let template_vertices: Vec<GpuTemplateVertex> = vertices
        .iter()
        .map(|v| GpuTemplateVertex {
            pos: [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32, 1.0],
            norm: [v.norm.x as f32, v.norm.y as f32, v.norm.z as f32, 0.0],
            color: [v.color.x as f32, v.color.y as f32, v.color.z as f32, 1.0],
        })
        .collect();
    trace!(
        "gpu_transform_vertices: Template vertices converted, count: {}",
        template_vertices.len()
    );

    // Convert transforms to GPU format
    let gpu_transforms: Vec<GpuTransform> = transforms
        .iter()
        .map(|t| {
            let inv_transpose = t
                .try_inverse()
                .map(|inv| inv.transpose())
                .unwrap_or_else(glm::DMat4::identity);
            GpuTransform {
                transform: mat_to_f32_array(t),
                inverse_transpose: mat_to_f32_array(&inv_transpose),
            }
        })
        .collect();
    trace!(
        "gpu_transform_vertices: Transforms converted, count: {}",
        gpu_transforms.len()
    );

    // Create GPU buffers
    let template_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Template Vertices Buffer"),
        contents: bytemuck::cast_slice(&template_vertices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    trace!("gpu_transform_vertices: Template buffer created");

    let transforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Transforms Buffer"),
        contents: bytemuck::cast_slice(&gpu_transforms),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    trace!("gpu_transform_vertices: Transforms buffer created");

    // Create output buffer for transformed vertices
    let num_output_vertices = template_vertices.len() * transforms.len();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Output Vertices Buffer"),
        size: (std::mem::size_of::<GpuOutputVertex>() * num_output_vertices) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    trace!(
        "gpu_transform_vertices: Output buffer created, size: {}, expected vertices: {}",
        output_buffer.size(),
        num_output_vertices
    );

    // Load shader code
    let shader_code = std::include_str!("transform_mesh.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Transform Mesh Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });
    trace!("gpu_transform_vertices: Shader module created");

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Transform Bind Group Layout"),
    });
    trace!("gpu_transform_vertices: Bind group layout created");

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: template_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: transforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
        label: Some("Transform Bind Group"),
    });
    trace!("gpu_transform_vertices: Bind group created");

    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Transform Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    trace!("gpu_transform_vertices: Pipeline layout created");

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Transform Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    trace!("gpu_transform_vertices: Compute pipeline created");

    // Create a staging buffer to read the results back from the GPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    trace!(
        "gpu_transform_vertices: Staging buffer created, size: {}",
        staging_buffer.size()
    );

    // Create command encoder and compute pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    trace!("gpu_transform_vertices: Command encoder created");
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((num_output_vertices as u32).div_ceil(64), 1, 1); // 64 workgroup size
    }
    trace!("gpu_transform_vertices: Compute pass completed");

    // Copy the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
    trace!("gpu_transform_vertices: Copy command issued");

    // Submit the command encoder
    let command_buffer = encoder.finish();
    queue.submit(Some(command_buffer));
    trace!("gpu_transform_vertices: Command buffer submitted");

    // Ensure the queue processes all commands before mapping
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Read the results back from the staging buffer
    assert_eq!(
        output_buffer.size(),
        staging_buffer.size(),
        "Buffer sizes must match"
    );

    let buffer_slice = staging_buffer.slice(..);
    trace!("gpu_transform_vertices: Buffer slice created");
    let (sender, receiver) = std::sync::mpsc::channel();
    trace!("gpu_transform_vertices: Channel created");

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        trace!(
            "gpu_transform_vertices: map_async callback called with result: {:?}",
            result.is_ok()
        );
        // Only send if the receiver is still around (ignore send errors)
        let _ = sender.send(result);
    });
    trace!("gpu_transform_vertices: map_async called");

    // Wait for the mapping to complete using new wgpu 27 syntax (single poll)
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Use timeout to prevent indefinite blocking
    match receiver.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => {
            trace!("gpu_transform_vertices: Buffer mapping succeeded");
            let data = buffer_slice.get_mapped_range();
            let result: Vec<GpuOutputVertex> = bytemuck::cast_slice(&data).to_vec();
            drop(data); // Drop the mapped range before unmap
            staging_buffer.unmap();
            trace!("gpu_transform_vertices: Buffer unmapped");

            let vertices: Vec<Vertex> = result
                .iter()
                .map(|v| Vertex {
                    pos: glm::DVec3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64),
                    norm: glm::DVec3::new(v.norm[0] as f64, v.norm[1] as f64, v.norm[2] as f64),
                    color: glm::DVec3::new(v.color[0] as f64, v.color[1] as f64, v.color[2] as f64),
                })
                .collect();
            trace!(
                "gpu_transform_vertices: Successfully returning {} vertices",
                vertices.len()
            );
            Ok(vertices)
        }
        Ok(Err(e)) => {
            log::error!("gpu_transform_vertices: Buffer mapping failed: {:?}", e);
            Err(format!("Buffer mapping failed: {:?}", e).into())
        }
        Err(_) => {
            log::error!("gpu_transform_vertices: Buffer mapping timed out");
            Err("Buffer mapping timed out".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use crate::surface::Surface;
    use nalgebra_glm as glm;

    #[test]
    fn test_gpu_transform_vertices() {
        trace!("test_gpu_transform_vertices: Starting test");

        // This test requires a GPU and might not run in all CI environments.
        // It's here for local testing.
        let (device, queue) = match create_wgpu_device() {
            Ok(dq) => {
                trace!("test_gpu_transform_vertices: Created wgpu device and queue");
                dq
            }
            Err(_) => {
                trace!(
                    "test_gpu_transform_vertices: Skipping GPU test: could not create wgpu device."
                );
                return;
            }
        };

        // 1. Define a surface (e.g., a plane)
        let surface = Surface::Plane {
            normal: glm::vec3(0.0, 0.0, 1.0),
            mat_i: glm::identity(),
        };
        trace!("test_gpu_transform_vertices: Created surface");

        // 2. Define some vertices on that surface
        let vertices = vec![
            Vertex {
                pos: glm::vec3(1.0, 2.0, 0.0),
                norm: glm::vec3(0.0, 0.0, 1.0),
                color: glm::vec3(1.0, 0.0, 0.0),
            },
            Vertex {
                pos: glm::vec3(3.0, 4.0, 0.0),
                norm: glm::vec3(0.0, 0.0, 1.0),
                color: glm::vec3(0.0, 1.0, 0.0),
            },
        ];
        trace!(
            "test_gpu_transform_vertices: Created {} test vertices",
            vertices.len()
        );

        // 3. Define a simple translation transform
        let transforms = vec![glm::translation(&glm::vec3(1.0, 1.0, 0.0))];
        trace!(
            "test_gpu_transform_vertices: Created {} test transforms",
            transforms.len()
        );

        // 4. Transform vertices using GPU
        trace!("test_gpu_transform_vertices: Starting GPU transformation");
        let transformed_vertices = pollster::block_on(gpu_transform_vertices(
            &device,
            &queue,
            &vertices,
            &surface,
            &transforms,
        ))
        .expect("GPU transformation failed");
        trace!(
            "test_gpu_transform_vertices: GPU transformation completed, got {} vertices",
            transformed_vertices.len()
        );

        // 5. Check if the transformed vertices match expectations
        assert_eq!(transformed_vertices.len(), 2);
        assert!((transformed_vertices[0].pos - glm::vec3(2.0, 3.0, 0.0)).norm() < 1e-6);
        assert!((transformed_vertices[1].pos - glm::vec3(4.0, 5.0, 0.0)).norm() < 1e-6);
        trace!("test_gpu_transform_vertices: Position validation passed");

        // Normals should remain unchanged for a simple translation
        assert!((transformed_vertices[0].norm - glm::vec3(0.0, 0.0, 1.0)).norm() < 1e-6);
        assert!((transformed_vertices[1].norm - glm::vec3(0.0, 0.0, 1.0)).norm() < 1e-6);
        trace!("test_gpu_transform_vertices: Normal validation passed");

        // Colors should match original vertices
        assert!((transformed_vertices[0].color - glm::vec3(1.0, 0.0, 0.0)).norm() < 1e-6);
        assert!((transformed_vertices[1].color - glm::vec3(0.0, 1.0, 0.0)).norm() < 1e-6);
        trace!("test_gpu_transform_vertices: Color validation passed");

        trace!("test_gpu_transform_vertices: Test completed successfully");
    }
}

/// Triangulate faces using GPU acceleration with a pre-existing GPU device
/// This function reuses GPU resources for better performance and resource management
pub fn triangulate_faces_with_device(
    s: &StepFile,
    face_tasks: &Vec<crate::triangulate::FaceTask>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (Mesh, Stats) {
    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    // Track statistics
    let total_faces = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let total_panics = AtomicUsize::new(0);

    let mesh = face_tasks
        .par_iter()
        .filter_map(|task| {
            total_faces.fetch_add(1, Ordering::Relaxed);

            // Get the face from the STEP file
            let face = match s.entity(task.face_id) {
                Some(f) => f,
                None => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            // Get the surface for this face (kept for API compatibility, but not used in current implementation)
            let _surface = match cpu_get_surface(s, face.face_geometry) {
                Ok(surf) => surf,
                Err(_) => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            // Create empty vectors to store results
            let mut verts = Vec::new();
            let mut triangles = Vec::new();

            // Initialize stats for the call
            let mut stats = Stats::default();

            // Call the advanced_face_to_mesh function with the proper signature
            match cpu_advanced_face_to_mesh(s, task.face_id, &mut verts, &mut triangles, &mut stats)
            {
                Ok(()) => {
                    if verts.is_empty() {
                        return Some(Mesh::default());
                    }

                    let mut transformed_mesh = Mesh::default();

                    // Use GPU only for transformations if there are transforms
                    let final_verts = if !task.transforms.is_empty() {
                        // Use new GPU transformation function instead of CPU-only approach
                        pollster::block_on(gpu_transform_vertices(
                            device,
                            queue,
                            &verts,
                            &_surface,
                            &task.transforms,
                        ))
                        .unwrap_or_else(|e| {
                            trace!("GPU transformation failed: {}", e);
                            total_errors.fetch_add(1, Ordering::Relaxed);
                            Vec::new()
                        })
                    } else {
                        // No transforms, just use original vertices and apply color
                        verts
                            .iter()
                            .map(|v| Vertex {
                                pos: v.pos,
                                norm: v.norm,
                                color: task.color,
                            })
                            .collect()
                    };

                    transformed_mesh.verts = final_verts;

                    // Replicate triangles for each transform
                    let num_template_verts = verts.len() as u32;
                    let num_transforms = if task.transforms.is_empty() {
                        1
                    } else {
                        task.transforms.len()
                    };

                    for i in 0..num_transforms {
                        let v_offset = i as u32 * num_template_verts;
                        for t in &triangles {
                            let mut tri = *t;
                            tri.verts.add_scalar_mut(v_offset);
                            transformed_mesh.triangles.push(tri);
                        }
                    }

                    // Flip normals if needed
                    if task.flip_normal {
                        for v in &mut transformed_mesh.verts {
                            v.norm = -v.norm;
                        }
                    }

                    Some(transformed_mesh)
                }
                Err(_) => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .reduce(Mesh::default, Mesh::combine);

    let stats = Stats {
        num_shells: 0, // This would need to be tracked separately
        num_faces: total_faces.load(Ordering::Relaxed),
        num_errors: total_errors.load(Ordering::Relaxed),
        num_panics: total_panics.load(Ordering::Relaxed),
    };

    (mesh, stats)
}

// New batch function to process all faces from multiple files in a single GPU operation
pub fn triangulate_multiple_files_batch(
    files_and_tasks: &[(StepFile, Vec<crate::triangulate::FaceTask>)],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<(Mesh, Stats)> {
    if files_and_tasks.is_empty() {
        return vec![];
    }

    // Process each file separately using the shared device, but at least we avoid device creation overhead
    files_and_tasks
        .iter()
        .map(|(step_file, face_tasks)| {
            triangulate_faces_with_device(step_file, face_tasks, device, queue)
        })
        .collect()
}
