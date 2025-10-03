use nalgebra_glm as glm;
use step::step_file::StepFile;
use wgpu::util::DeviceExt;

use crate::mesh::{Mesh, Vertex};
use crate::stats::Stats;
use crate::surface::Surface;

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
pub struct GpuInputVertex {
    pos: [f32; 4],   // 3D position (xyz) + padding
    norm: [f32; 4],  // normal (xyz) + padding
    color: [f32; 4], // color (rgb) + padding
}

// Structure for output (UV coordinates)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuOutputVertex {
    uv: [f32; 2],       // UV coordinates
    norm: [f32; 4],     // normal (xyz) + padding
    _padding: [f32; 2], // padding
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
        Surface::Plane { normal, mat_i } => GpuSurface {
            surface_type: SURFACE_TYPE_PLANE,
            mat_i: mat_to_f32_array(mat_i),
            axis: vec3_to_f32_array(normal),
            radius: 0.0,
            major_radius: 0.0,
            minor_radius: 0.0,
            mat: [[0.0; 4]; 4], // identity will be set if needed
            z_min: 0.0,
            z_max: 0.0,
            angle: 0.0,
            location: [0.0, 0.0, 0.0], // will be set if needed
            _p0: 0,
            _p1: 0.0,
            _p2: 0.0,
        },
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

#[cfg(feature = "wgpu")]
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

// GPU compute pass to lower 3D points to 2D UV coordinates
#[cfg(feature = "wgpu")]
pub async fn gpu_lower_vertices(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertices: &[Vertex],
    surface: &Surface,
) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    // Convert surface to GPU format
    let gpu_surface = to_gpu_surface(surface);

    // Convert input vertices to GPU format
    let input_vertices: Vec<GpuInputVertex> = vertices
        .iter()
        .map(|v| GpuInputVertex {
            pos: [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32, 1.0],
            norm: [v.norm.x as f32, v.norm.y as f32, v.norm.z as f32, 0.0],
            color: [v.color.x as f32, v.color.y as f32, v.color.z as f32, 1.0],
        })
        .collect();

    // Create GPU buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Input Vertices Buffer"),
        contents: bytemuck::cast_slice(&input_vertices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let surface_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Surface Buffer"),
        contents: bytemuck::cast_slice(&[gpu_surface]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create output buffer for UV coordinates
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Output UV Buffer"),
        size: (std::mem::size_of::<GpuOutputVertex>() * vertices.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader code
    let shader_code = std::include_str!("surface_ops.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surface Operations Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
        label: Some("Lowering Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: surface_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
        label: Some("Lowering Bind Group"),
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lowering Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lowering Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("lowering_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and compute pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((vertices.len() as u32 + 63) / 64, 1, 1); // 64 workgroup size
    }

    // Submit the command encoder
    queue.submit(Some(encoder.finish()));

    // For now, just return an empty result to avoid blocking on the async operation
    // In a proper implementation, we would wait for the GPU to finish and read back results
    Ok(Vec::new())
}

// Structure for GPU transforms
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTransform {
    transform: [[f32; 4]; 4],
}

// GPU compute pass to raise 2D points to 3D and apply transformations
#[cfg(feature = "wgpu")]
pub async fn gpu_raise_and_transform(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    template_uvs: &[(f64, f64)],
    template_normals: &[glm::DVec3],
    surface: &Surface,
    transforms: &[glm::DMat4],
) -> Result<Vec<Vertex>, Box<dyn std::error::Error>> {
    let gpu_surface = to_gpu_surface(surface);

    // Convert UV coordinates to GPU format
    let gpu_uvs: Vec<[f32; 4]> = template_uvs
        .iter()
        .map(|uv| [uv.0 as f32, uv.1 as f32, 0.0, 0.0])
        .collect();

    // Convert normals to GPU format
    let gpu_normals: Vec<[f32; 4]> = template_normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32, 0.0])
        .collect();

    // Convert transforms to GPU format
    let gpu_transforms: Vec<GpuTransform> = transforms
        .iter()
        .map(|t| GpuTransform {
            transform: mat_to_f32_array(t),
        })
        .collect();

    // Create GPU buffers
    let surface_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Surface Buffer"),
        contents: bytemuck::cast_slice(&[gpu_surface]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let template_uvs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Template UVs Buffer"),
        contents: bytemuck::cast_slice(&gpu_uvs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let template_normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Template Normals Buffer"),
        contents: bytemuck::cast_slice(&gpu_normals),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let transforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Transforms Buffer"),
        contents: bytemuck::cast_slice(&gpu_transforms),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer for final vertices
    let num_output_vertices = template_uvs.len() * transforms.len();
    let output_vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Vertices Buffer"),
        size: (std::mem::size_of::<GpuInputVertex>() * num_output_vertices) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader code
    let shader_code = std::include_str!("surface_ops.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surface Raise and Transform Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Raise and Transform Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: surface_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: template_uvs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: template_normals_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: transforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_vertices_buffer.as_entire_binding(),
            },
        ],
        label: Some("Raise and Transform Bind Group"),
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Raise and Transform Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Raise and Transform Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("raising_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and compute pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((num_output_vertices as u32 + 63) / 64, 1, 1); // 64 workgroup size
    }

    // Submit the command encoder
    let command_buffer = encoder.finish();
    queue.submit(Some(command_buffer));

    // Since we're not currently reading back the results for this function in the shader,
    // let's implement similar pattern for now, but in a real implementation you'd map
    // the output buffer and read back the results
    
    // For now, return an empty vector to avoid breaking compilation
    // This would be completed in a full implementation
    Ok(Vec::new())
}

#[cfg(feature = "wgpu")]
use crate::triangulate::{
    advanced_face_to_mesh as cpu_advanced_face_to_mesh, get_surface as cpu_get_surface,
};
#[cfg(feature = "wgpu")]
use glm::DVec4;
#[cfg(feature = "wgpu")]
use std::sync::atomic::{AtomicUsize, Ordering};

// Main orchestration function
#[cfg(feature = "wgpu")]
pub fn triangulate_faces(
    s: &StepFile,
    face_tasks: &Vec<crate::triangulate::FaceTask>,
) -> (Mesh, Stats) {
    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    // Initialize wgpu
    let (_device, _queue) = create_wgpu_device().expect("Failed to create wgpu device");

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

            // Get the surface for this face
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
                    // Apply transforms to the resulting mesh
                    let mut transformed_mesh = Mesh::default();
                    for mat in &task.transforms {
                        let v_offset = transformed_mesh.verts.len() as u32;

                        // Apply transform to each vertex
                        for v in &verts {
                            let p_h = DVec4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                            let pos = (mat * p_h).xyz();

                            let n_h = DVec4::new(v.norm.x, v.norm.y, v.norm.z, 0.0);
                            let norm = (mat * n_h).xyz().normalize();

                            transformed_mesh.verts.push(Vertex {
                                pos,
                                norm,
                                color: task.color,
                            });
                        }

                        // Add transformed triangles with offset indices
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
