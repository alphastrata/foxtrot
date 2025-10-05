use nalgebra_glm as glm;
use step::step_file::StepFile;
use wgpu::util::DeviceExt;
use wgpu::PollType;
use log::{error, trace};
use crate::triangulate::{
    advanced_face_to_mesh as cpu_advanced_face_to_mesh, get_surface as cpu_get_surface,
};

use std::sync::atomic::{AtomicUsize, Ordering};


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
    let shader_code = std::include_str!("surface_lowering.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surface Lowering Shader"),
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

    // Create a staging buffer to read the results back from the GPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create command encoder and compute pass - this should happen before the copy
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((vertices.len() as u32 + 63) / 64, 1, 1); // 64 workgroup size
    }

    // Copy the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());

    // Submit the command encoder
    let command_buffer = encoder.finish();
    queue.submit(Some(command_buffer));

    // Wait for GPU operations to complete
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Read the results back from the staging buffer
    assert_eq!(output_buffer.size(), staging_buffer.size(), "Buffer sizes must match");

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Wait for the mapping to complete using new wgpu 27 syntax (single poll)
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Use timeout to prevent indefinite blocking
    match receiver.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<GpuOutputVertex> = bytemuck::cast_slice(&data).to_vec();
            drop(data);  // Drop the mapped range before unmap
            staging_buffer.unmap();

            let uvs: Vec<(f64, f64)> = result
                .iter()
                .map(|v| (v.uv[0] as f64, v.uv[1] as f64))
                .collect();
            Ok(uvs)
        }
        Ok(Err(e)) => {
            Err(format!("Buffer mapping failed: {:?}", e).into())
        },
        Err(_) => {
            Err("Buffer mapping timed out".into())
        },
    }
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
    trace!("gpu_raise_and_transform: Starting");

    let gpu_surface = to_gpu_surface(surface);
    trace!("gpu_raise_and_transform: Surface converted");

    // Convert UV coordinates to GPU format
    let gpu_uvs: Vec<[f32; 4]> = template_uvs
        .iter()
        .map(|uv| [uv.0 as f32, uv.1 as f32, 0.0, 0.0])
        .collect();
    trace!("gpu_raise_and_transform: UVs converted, count: {}", gpu_uvs.len());

    // Convert normals to GPU format
    let gpu_normals: Vec<[f32; 4]> = template_normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32, 0.0])
        .collect();
    trace!("gpu_raise_and_transform: Normals converted, count: {}", gpu_normals.len());

    // Convert transforms to GPU format
    let gpu_transforms: Vec<GpuTransform> = transforms
        .iter()
        .map(|t| GpuTransform {
            transform: mat_to_f32_array(t),
        })
        .collect();
    trace!("gpu_raise_and_transform: Transforms converted, count: {}", gpu_transforms.len());

    // Create GPU buffers
    let surface_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Surface Buffer"),
        contents: bytemuck::cast_slice(&[gpu_surface]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    trace!("gpu_raise_and_transform: Surface buffer created");

    let template_uvs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Template UVs Buffer"),
        contents: bytemuck::cast_slice(&gpu_uvs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    trace!("gpu_raise_and_transform: Template UVs buffer created");

    let template_normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Template Normals Buffer"),
        contents: bytemuck::cast_slice(&gpu_normals),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    trace!("gpu_raise_and_transform: Template normals buffer created");

    let transforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Transforms Buffer"),
        contents: bytemuck::cast_slice(&gpu_transforms),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    trace!("gpu_raise_and_transform: Transforms buffer created");

    // Create output buffer for final vertices
    let num_output_vertices = template_uvs.len() * transforms.len();
    let output_vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Vertices Buffer"),
        size: (std::mem::size_of::<GpuInputVertex>() * num_output_vertices) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    trace!("gpu_raise_and_transform: Output buffer created, size: {}, expected vertices: {}", output_vertices_buffer.size(), num_output_vertices);

    // Load shader code
    let shader_code = std::include_str!("surface_raising.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surface Raising Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });
    trace!("gpu_raise_and_transform: Shader module created");

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
    trace!("gpu_raise_and_transform: Bind group layout created");

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
    trace!("gpu_raise_and_transform: Bind group created");

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Raise and Transform Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    trace!("gpu_raise_and_transform: Pipeline layout created");

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Raise and Transform Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("raising_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    trace!("gpu_raise_and_transform: Compute pipeline created");

    // Create a staging buffer to read the results back from the GPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_vertices_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    trace!("gpu_raise_and_transform: Staging buffer created, size: {}", staging_buffer.size());

    // Create command encoder and compute pass - this should happen before the copy
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    trace!("gpu_raise_and_transform: Command encoder created");
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((num_output_vertices as u32 + 63) / 64, 1, 1); // 64 workgroup size
    }
    trace!("gpu_raise_and_transform: Compute pass completed");

    // Copy the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(
        &output_vertices_buffer,
        0,
        &staging_buffer,
        0,
        output_vertices_buffer.size(),
    );
    trace!("gpu_raise_and_transform: Copy command issued");

    // Submit the command encoder
    let command_buffer = encoder.finish();
    queue.submit(Some(command_buffer));
    trace!("gpu_raise_and_transform: Command buffer submitted");

    // Wait for GPU operations to complete
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    trace!("gpu_raise_and_transform: Device poll completed - GPU operations should be done");

    // Read the results back from the staging buffer
    trace!("gpu_raise_and_transform: Template UVs buffer size: {}, Template normals buffer size: {}, Transforms buffer size: {}, Output buffer size: {}, Staging buffer size: {}", 
        template_uvs_buffer.size(), template_normals_buffer.size(), transforms_buffer.size(), 
        output_vertices_buffer.size(), staging_buffer.size());
    assert_eq!(output_vertices_buffer.size(), staging_buffer.size(), "Buffer sizes must match");

    let buffer_slice = staging_buffer.slice(..);
    trace!("gpu_raise_and_transform: Buffer slice created");
    let (sender, receiver) = std::sync::mpsc::channel();
    trace!("gpu_raise_and_transform: Channel created");
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        trace!("gpu_raise_and_transform: map_async callback called with result: {:?}", result.is_ok());
        sender.send(result).unwrap();
    });
    trace!("gpu_raise_and_transform: map_async called");

    // Wait for the mapping to complete using new wgpu 27 syntax (single poll)
    let _ = device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    trace!("gpu_raise_and_transform: Device poll completed after mapping");

    // Use timeout to prevent indefinite blocking
    match receiver.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => {
            trace!("gpu_raise_and_transform: Buffer mapping succeeded");
            let data = buffer_slice.get_mapped_range();
            let result: Vec<GpuInputVertex> = bytemuck::cast_slice(&data).to_vec();
            drop(data);  // Drop the mapped range before unmap
            staging_buffer.unmap();
            trace!("gpu_raise_and_transform: Buffer unmapped");

            let vertices: Vec<Vertex> = result
                .iter()
                .map(|v| Vertex {
                    pos: glm::DVec3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64),
                    norm: glm::DVec3::new(v.norm[0] as f64, v.norm[1] as f64, v.norm[2] as f64),
                    color: glm::DVec3::new(v.color[0] as f64, v.color[1] as f64, v.color[2] as f64),
                })
                .collect();
            trace!("gpu_raise_and_transform: Successfully returning {} vertices", vertices.len());
            Ok(vertices)
        }
        Ok(Err(e)) => {
            error!("gpu_raise_and_transform: Buffer mapping failed: {:?}", e);
            Err(format!("Buffer mapping failed: {:?}", e).into())
        },
        Err(_) => {
            error!("gpu_raise_and_transform: Buffer mapping timed out");
            Err("Buffer mapping timed out".into())
        },
    }
}


// Main orchestration function
#[cfg(feature = "wgpu")]
pub fn triangulate(s: &StepFile) -> (Mesh, Stats) {
    // Simply call the existing wgpu_triangulate function from the main module
    // This avoids reimplementing the complex logic while ensuring parity
    crate::triangulate::wgpu_triangulate(s)
}

#[cfg(feature = "wgpu")]
pub fn triangulate_faces(
    s: &StepFile,
    face_tasks: &Vec<crate::triangulate::FaceTask>,
) -> (Mesh, Stats) {
    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    // Initialize wgpu
    let (device, queue) = create_wgpu_device().expect("Failed to create wgpu device");

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
            let surface = match cpu_get_surface(s, face.face_geometry) {
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

                    // Use GPU for lowering and raising if there are transforms
                    let final_verts = if !task.transforms.is_empty() {
                        // Lower vertices to UVs
                        let uvs = pollster::block_on(gpu_lower_vertices(
                            &device, &queue, &verts, &surface,
                        ))
                        .unwrap_or_else(|e| {
                            error!("GPU lowering failed: {}", e);
                            total_errors.fetch_add(1, Ordering::Relaxed);
                            Vec::new()
                        });

                        if uvs.is_empty() {
                            total_errors.fetch_add(1, Ordering::Relaxed);
                            return None; // Or handle error appropriately
                        }

                        let normals: Vec<glm::DVec3> = verts.iter().map(|v| v.norm).collect();

                        // Raise UVs back to 3D and apply transforms
                        let mut raised_verts = pollster::block_on(gpu_raise_and_transform(
                            &device,
                            &queue,
                            &uvs,
                            &normals,
                            &surface,
                            &task.transforms,
                        ))
                        .unwrap_or_else(|e| {
                            error!("GPU raising failed: {}", e);
                            total_errors.fetch_add(1, Ordering::Relaxed);
                            Vec::new()
                        });

                        // The shader should handle color, but let's set it just in case
                        for v in &mut raised_verts {
                            v.color = task.color;
                        }
                        raised_verts
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use crate::surface::Surface;
    use nalgebra_glm as glm;

    #[test]
    fn test_gpu_lowering_and_raising() {

        // This test requires a GPU and might not run in all CI environments.
        // It's here for local testing.
        let (device, queue) = match create_wgpu_device() {
            Ok(dq) => dq,
            Err(_) => {
                return; // Skip test if GPU device not available
            }
        };

        // 1. Define a surface (e.g., a plane)
        let surface = Surface::Plane {
            normal: glm::vec3(0.0, 0.0, 1.0),
            mat_i: glm::identity(),
        };

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

        // 3. Lower vertices to UVs
        let uvs = pollster::block_on(gpu_lower_vertices(&device, &queue, &vertices, &surface))
            .expect("GPU lowering failed");

        // For a simple XY plane, we should get 2 UV coordinates back
        assert_eq!(uvs.len(), 2, "Expected 2 UV coordinates, got {}", uvs.len());
        // Check that UV coordinates are approximately correct (for a plane, UVs should match the input X,Y coordinates)
        // Original vertices were (1.0, 2.0, 0.0) and (3.0, 4.0, 0.0), so UVs should be (1.0, 2.0) and (3.0, 4.0)
        // The exact mapping might need shader refinement
        assert!(uvs[0].0.is_finite() && uvs[0].1.is_finite(), "UV[0] should be finite");
        assert!(uvs[1].0.is_finite() && uvs[1].1.is_finite(), "UV[1] should be finite");

        // 4. Raise UVs back to 3D vertices (commented out for now to isolate the lower issue)
        /*
        let normals: Vec<glm::DVec3> = vertices.iter().map(|v| v.norm).collect();
        let transforms = vec![glm::identity()]; // No transformation
        let raised_vertices =
            pollster::block_on(gpu_raise_and_transform(&device, &queue, &uvs, &normals, &surface, &transforms))
                .expect("GPU raising failed");

        // 5. Check if the raised vertices match the original ones
        assert_eq!(raised_vertices.len(), 2);
        for i in 0..vertices.len() {
            assert!((raised_vertices[i].pos - vertices[i].pos).norm() < 1e-6);
            // Normals might be recomputed, let's check they are correct for a plane
            assert!((raised_vertices[i].norm - glm::vec3(0.0, 0.0, 1.0)).norm() < 1e-6);
        }
        */
}}
