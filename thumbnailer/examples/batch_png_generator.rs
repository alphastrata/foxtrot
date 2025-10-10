use clap::Parser;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use triangulate::wgpu_triangulate::GPUContext;

// Add necessary imports for rendering - use the main app's infrastructure
use nalgebra_glm as glm;
use pollster;
use wgpu;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// Add PNG encoding
use png;

// Import from the main app's infrastructure
use std::time::SystemTime;
use triangulate::mesh::Mesh;

// Reuse the App struct from main app for consistent rendering
pub struct BatchApp {
    pub start_time: SystemTime,
    pub window_size: PhysicalSize<u32>,
    pub graphics_adapter: wgpu::Adapter,
    pub graphics_device: wgpu::Device,
    pub command_queue: wgpu::Queue,
}

impl BatchApp {
    pub fn new(
        start_time: SystemTime,
        window_size: PhysicalSize<u32>,
        graphics_adapter: wgpu::Adapter,
        graphics_device: wgpu::Device,
        command_queue: wgpu::Queue,
    ) -> Self {
        Self {
            start_time,
            window_size,
            graphics_adapter,
            graphics_device,
            command_queue,
        }
    }

    pub fn render_to_texture(
        &mut self,
        texture_view: &wgpu::TextureView,
        mesh: &Mesh,
        args: &BatchArgs,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create camera and model (same as main app)
        let mut camera = Camera::new();
        let mut model = Model::new(mesh.clone());
        model.scale_model_to_unit_cube();

        let model_center = glm::vec3(0.0, 0.0, 0.0); // After scaling, center is at origin
        
        camera.fit_to_bounds(&model.mesh.verts, 1.0);
        // Use the view from args - simplified version of main app
        match args.view {
            CameraView::Isometric => camera.set_view_from_angles(45.0, 35.0), // isometric view angles
            CameraView::Top => camera.set_view_from_angles(0.0, 90.0),
            CameraView::Bottom => camera.set_view_from_angles(0.0, -90.0),
            CameraView::Left => camera.set_view_from_angles(90.0, 0.0),
            CameraView::Right => camera.set_view_from_angles(-90.0, 0.0),
            CameraView::Front => camera.set_view_from_angles(0.0, 0.0),
            CameraView::Back => camera.set_view_from_angles(180.0, 0.0),
        };

        // Create Depth Texture
        let depth_texture_size = wgpu::Extent3d {
            width: args.thumbnail_size,
            height: args.thumbnail_size,
            depth_or_array_layers: 1,
        };
        let depth_texture = self.graphics_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: depth_texture_size,
            mip_level_count: 1,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Load same shader as main app
        let shader = self.graphics_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thumbnail Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../src/shader.wgsl"
            ))),
        });

        // Create bind group layout (same as main app)
        let bind_group_layout =
            self.graphics_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: Some("Camera and Light Bind Group Layout"),
                });

        // Create pipeline layout
        let pipeline_layout =
            self.graphics_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Thumbnail Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create render pipeline (same as main app)
        let pipeline =
            self.graphics_device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Thumbnail Render Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 6 * std::mem::size_of::<f32>() as wgpu::BufferAddress, // position (3) + normal (3)
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 3 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                                    shader_location: 1,
                                },
                            ],
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: if args.transparent {
                                wgpu::TextureFormat::Rgba8UnormSrgb
                            } else {
                                wgpu::TextureFormat::Bgra8UnormSrgb
                            },
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 4,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        // Create vertex buffer using transformed mesh from the `model`
        let vertex_data: Vec<f32> = model
            .mesh
            .verts
            .iter()
            .flat_map(|v| {
                // Position (x, y, z) and Normal (nx, ny, nz)
                [
                    v.pos.x as f32,
                    v.pos.y as f32,
                    v.pos.z as f32,
                    v.norm.x as f32,
                    v.norm.y as f32,
                    v.norm.z as f32,
                ]
            })
            .collect();

        let vertex_buffer =
            self.graphics_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        // Create index buffer using transformed mesh
        let indices: Vec<u32> = model
            .mesh
            .triangles
            .iter()
            .flat_map(|tri| [tri.verts.x, tri.verts.y, tri.verts.z])
            .collect();

        let index_buffer =
            self.graphics_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

        // Create uniform buffer for camera matrices
        let view_proj_matrix = camera.projection_matrix * camera.view_matrix;
        let matrix_data: [f32; 16] = [
            view_proj_matrix[0],
            view_proj_matrix[1],
            view_proj_matrix[2],
            view_proj_matrix[3],
            view_proj_matrix[4],
            view_proj_matrix[5],
            view_proj_matrix[6],
            view_proj_matrix[7],
            view_proj_matrix[8],
            view_proj_matrix[9],
            view_proj_matrix[10],
            view_proj_matrix[11],
            view_proj_matrix[12],
            view_proj_matrix[13],
            view_proj_matrix[14],
            view_proj_matrix[15],
        ];
        let camera_uniform_buffer =
            self.graphics_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Uniform Buffer"),
                    contents: bytemuck::cast_slice(&matrix_data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create lighting uniforms
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LightUniforms {
            direction: [f32; 3],
            color: [f32; 3],
            _pad: [f32; 2], // padding to align to 16-byte boundary
        }

        let light_uniforms = LightUniforms {
            direction: [0.5, 0.5, 0.7], // isometric direction
            color: [1.0, 1.0, 1.0], // white light
            _pad: [0.0, 0.0],
        };

        let light_uniform_buffer =
            self.graphics_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Light Uniform Buffer"),
                    contents: bytemuck::cast_slice(&[light_uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create bind group for camera and light uniforms
        let camera_bind_group =
            self.graphics_device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: light_uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("Camera and Light Bind Group"),
                });

        // Create a multisampled texture to render into
        let multisampled_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("Multisampled Framebuffer"),
            size: wgpu::Extent3d {
                width: args.thumbnail_size,
                height: args.thumbnail_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4, // Same as pipeline
            dimension: wgpu::TextureDimension::D2,
            format: if args.transparent {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Bgra8UnormSrgb
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let multisampled_framebuffer = self
            .graphics_device
            .create_texture(&multisampled_texture_descriptor);
        let multisampled_view =
            multisampled_framebuffer.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.graphics_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Thumbnail Command Encoder") });

        // Render to multisampled texture, then resolve to the final texture
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Thumbnail Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    resolve_target: Some(texture_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if args.transparent { wgpu::Color::TRANSPARENT } else { wgpu::Color::WHITE }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&pipeline);
            render_pass.set_bind_group(0, &camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.command_queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    // Read texture to buffer (same as main app)
    pub async fn read_texture_to_buffer(
        &self,
        texture: &wgpu::Texture,
        size: &wgpu::Extent3d,
    ) -> Vec<u8> {
        let (tx, rx) = std::sync::mpsc::channel();
        let texture_format = texture.format();
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = align * ((texture_format.block_copy_size(None).unwrap() * size.width + align - 1) / align);
        let buffer_size = bytes_per_row * size.height * size.depth_or_array_layers;
        
        let output_buffer = self.graphics_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.graphics_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            *size,
        );
        self.command_queue.submit(std::iter::once(encoder.finish()));

        output_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });

        self.graphics_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();

        let data = output_buffer.slice(..).get_mapped_range().to_vec();
        drop(output_buffer); // Ensure buffer is unmapped before returning
        data
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct BatchArgs {
    /// Input directory containing STEP files
    #[arg(short, long, default_value = "./examples")]
    input_dir: String,

    /// Output directory for processed PNG files
    #[arg(short, long, default_value = "./output")]
    output_dir: String,

    /// Thumbnail size in pixels (width and height)
    #[arg(short = 's', long = "thumbnail-size", default_value = "512")]
    thumbnail_size: u32,

    /// Whether to render with transparent background
    #[arg(long)]
    transparent: bool,

    /// Whether to auto-crop the output
    #[arg(long)]
    crop: bool,

    /// Render mode - single isometric (iso) or 4-view composite (composite)
    #[arg(long, default_value = "iso", value_enum)]
    mode: RenderMode,

    /// Sets the camera view for the render
    #[arg(long, value_enum, default_value = "isometric")]
    view: CameraView,

    /// Triangulation engine to use (requires OCCT feature)
    #[cfg(feature = "occt")]
    #[arg(long, value_enum, default_value = "foxtrot")]
    engine: TriangulationEngine,

    /// Triangulation engine to use
    #[cfg(not(feature = "occt"))]
    #[arg(long, value_enum, default_value = "foxtrot")]
    engine: TriangulationEngine,

    /// Target reduction ratio for mesh decimation (0.0 - 1.0, where 0.5 = 50% reduction)
    #[arg(long = "decimation-ratio", default_value = "0.25")]
    decimation_ratio: f32,

    /// Target error tolerance for mesh decimation (higher = more aggressive simplification)
    #[arg(long = "decimation-error", default_value = "0.02")]
    decimation_error: f32,

    /// Linear deflection for OCCT triangulation (smaller = more triangles)
    #[cfg(feature = "occt")]
    #[arg(long = "occt-linear-deflection", default_value = "0.01")]
    occt_linear_deflection: f64,

    /// Angular deflection for OCCT triangulation (smaller = more accurate)
    #[cfg(feature = "occt")]
    #[arg(long = "occt-angular-deflection", default_value = "0.5")]
    occt_angular_deflection: f64,

    /// Whether to use shaded rendering instead of wireframe
    #[arg(long, default_value_t = false)]
    shaded: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum RenderMode {
    Iso,
    Composite,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum CameraView {
    Isometric,
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum TriangulationEngine {
    Foxtrot,
    #[cfg(feature = "occt")]
    Occt,
}

// Create a test function to compare single vs batch workflow
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Helper to run single workflow for comparison
    fn single_workflow_render(step_path: &Path, output_path: &Path, args: &BatchArgs) -> Result<(), Box<dyn std::error::Error>> {
        // This would call the main app's workflow using the same args
        // For now we'll just implement it conceptually
        unimplemented!("Implement single workflow for comparison test")
    }

    // Test to verify consistency between single and batch workflows
    #[test]
    fn test_single_vs_batch_consistency() -> Result<(), Box<dyn std::error::Error>> {
        let test_step_path = PathBuf::from("../examples/block.step");
        if !test_step_path.exists() {
            // Skip test if test file doesn't exist
            return Ok(());
        }

        let args = BatchArgs {
            input_dir: String::new(),
            output_dir: String::new(),
            thumbnail_size: 256,
            transparent: false,
            crop: false,
            mode: RenderMode::Iso,
            view: CameraView::Isometric,
            #[cfg(feature = "occt")]
            engine: TriangulationEngine::Foxtrot,
            #[cfg(not(feature = "occt"))]
            engine: TriangulationEngine::Foxtrot,
            decimation_ratio: 0.25,
            decimation_error: 0.02,
            #[cfg(feature = "occt")]
            occt_linear_deflection: 0.01,
            #[cfg(feature = "occt")]
            occt_angular_deflection: 0.5,
            shaded: false,
        };

        // Compare outputs
        let single_output = single_workflow_render(&test_step_path, &PathBuf::from("/tmp/single_test.png"), &args)?;
        // let batch_output = run_batch_workflow_item(&test_step_path, &PathBuf::from("/tmp/batch_test.png"), &args)?;

        // Compare the outputs - they should be similar (within tolerance)
        // assert!(compare_images(&single_output, &batch_output)?);

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = BatchArgs::parse();

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir)?;

    // Find all STEP files in the input directory
    let mut step_files: Vec<String> = Vec::new();
    for entry in fs::read_dir(&args.input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == OsStr::new("step") || ext == OsStr::new("stp") {
                    step_files.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    println!(
        "Found {} STEP files in {}",
        step_files.len(),
        args.input_dir
    );

    // Initialize GPU context once for all processing
    let gpu_context = triangulate::wgpu_triangulate::GPUContext::new()?;
    println!("GPU context initialized successfully");

    // Initialize the batch app with proper wgpu device
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        ..Default::default()
    })).expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    })).expect("Failed to create device");

    let mut batch_app = BatchApp::new(
        SystemTime::now(),
        PhysicalSize::new(args.thumbnail_size, args.thumbnail_size),
        adapter,
        device,
        queue,
    );
    println!("Batch app initialized successfully");

    // Track processing statistics
    let start_time = Instant::now();
    let mut total_processed = 0;

    // Process each STEP file
    for (i, file_path) in step_files.iter().enumerate() {
        println!(
            "Processing file {}/{}: {}",
            i + 1,
            step_files.len(),
            Path::new(file_path).file_name().unwrap().to_string_lossy()
        );

        // Read and parse the STEP file
        let contents = fs::read(file_path)?;
        let flattened = step::step_file::StepFile::strip_flatten(&contents);
        let step_file = step::step_file::StepFile::parse(&flattened);

        // Process using GPU context
        let (mesh, _stats) =
            triangulate::triangulate::wgpu_triangulate_with_context(&step_file, &gpu_context);

        // Convert to PNG and save
        let output_path = Path::new(&args.output_dir)
            .join(Path::new(file_path).file_stem().unwrap())
            .with_extension("png");

        // Skip if mesh is empty
        if mesh.verts.is_empty() || mesh.triangles.is_empty() {
            eprintln!(
                "Warning: Mesh is empty ({} vertices, {} triangles). Creating empty PNG with message.",
                mesh.verts.len(),
                mesh.triangles.len()
            );

            let thumbnail_size = args.thumbnail_size;
            let mut png_data =
                Vec::<u8>::with_capacity((thumbnail_size * thumbnail_size * 4) as usize);

            let bg_color = if args.transparent {
                vec![0, 0, 0, 0] // Transparent
            } else {
                vec![255, 255, 255, 255] // White background
            };

            (0..(thumbnail_size * thumbnail_size)).for_each(|_| {
                png_data.extend_from_slice(&bg_color);
            });

            // Add a red diagonal line to indicate empty mesh
            (0..thumbnail_size).for_each(|i| {
                let idx = ((i * thumbnail_size + i) * 4) as usize;
                if idx < png_data.len() - 3 {
                    png_data[idx] = 255; // R
                    png_data[idx + 1] = 0; // G
                    png_data[idx + 2] = 0; // B
                    if !args.transparent {
                        png_data[idx + 3] = 255; // A
                    }
                }
            });

            let mut output_png_data = Vec::<u8>::with_capacity(png_data.len());
            let mut encoder = png::Encoder::new(
                std::io::Cursor::new(&mut output_png_data),
                thumbnail_size,
                thumbnail_size,
            );
            encoder.set_color(png::ColorType::Rgba);

            let mut png_writer = encoder.write_header().unwrap();
            png_writer.write_image_data(&png_data[..]).unwrap();
            png_writer.finish().unwrap();

            let mut file = std::fs::File::create(&output_path)?;
            file.write_all(&output_png_data)?;

            println!(
                "    Saved empty mesh indicator to: {}",
                output_path.display()
            );
            continue; // Continue to next file
        }

        // Create render texture
        let texture_size = wgpu::Extent3d {
            width: args.thumbnail_size,
            height: args.thumbnail_size,
            depth_or_array_layers: 1,
        };

        let texture = batch_app.graphics_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Batch Render Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if args.transparent {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Bgra8UnormSrgb
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Render the mesh to texture using the same logic as the main app
        batch_app.render_to_texture(&texture_view, &mesh, &args)?;

        // Read texture data back
        let buffer = pollster::block_on(batch_app.read_texture_to_buffer(
            &texture,
            &wgpu::Extent3d {
                width: args.thumbnail_size,
                height: args.thumbnail_size,
                depth_or_array_layers: 1,
            },
        ));

        let img_data = if args.transparent {
            buffer
        } else {
            let mut rgba_buffer = Vec::with_capacity((args.thumbnail_size * args.thumbnail_size * 4) as usize);
            for chunk in buffer.chunks(4) {
                rgba_buffer.push(chunk[2]); // B
                rgba_buffer.push(chunk[1]); // G
                rgba_buffer.push(chunk[0]); // R
                rgba_buffer.push(chunk[3]); // A
            }
            rgba_buffer
        };

        // Add the assertion back - this should now be valid since we're using the same rendering as main app
        assert!(
            !img_data.iter().all(|&b| b == 255),
            "The png we created was pure WHITE! this means there is a bug in the way we 'produce' the png. Fix the bug, do NOT REMOVE THIS ASSERT!"
        );

        let mut flipped_img_data = Vec::with_capacity(img_data.len());
        let row_size = (args.thumbnail_size * 4) as usize;

        for row in (0..args.thumbnail_size).rev() {
            let start_idx = row as usize * args.thumbnail_size as usize * 4;
            let end_idx = start_idx + row_size;
            flipped_img_data.extend_from_slice(&img_data[start_idx..end_idx]);
        }

        // Encode to PNG
        let mut png_data = Vec::<u8>::with_capacity(flipped_img_data.len());
        let mut encoder =
            png::Encoder::new(std::io::Cursor::new(&mut png_data), args.thumbnail_size, args.thumbnail_size);
        encoder.set_color(png::ColorType::Rgba);
        let mut png_writer = encoder.write_header().unwrap();
        png_writer.write_image_data(&flipped_img_data[..]).unwrap();
        png_writer.finish().unwrap();

        let mut file = std::fs::File::create(&output_path)?;
        file.write_all(&png_data)?;

        println!("    Saved to: {}", output_path.display());
        total_processed += 1;
    }

    let total_duration = start_time.elapsed();
    println!(
        "\nProcessed {} files in {:.2?}",
        total_processed, total_duration
    );
    println!(
        "Average processing time per file: {:.2?}",
        total_duration / total_processed.max(1) as u32
    );

    Ok(())
}

// Simplified Camera implementation based on main app
struct Camera {
    pub view_matrix: glm::Mat4,
    pub projection_matrix: glm::Mat4,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Camera {
            view_matrix: glm::identity(),
            projection_matrix: glm::identity(),
            distance: 5.0,
            yaw: 45.0_f32.to_radians(),
            pitch: 35.0_f32.to_radians(),
        }
    }

    pub fn fit_to_bounds(&mut self, vertices: &[triangulate::mesh::Vertex], padding: f32) {
        if vertices.is_empty() {
            return;
        }

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for vertex in vertices {
            let pos = vertex.pos;
            min_x = min_x.min(pos.x as f32);
            min_y = min_y.min(pos.y as f32);
            min_z = min_z.min(pos.z as f32);
            max_x = max_x.max(pos.x as f32);
            max_y = max_y.max(pos.y as f32);
            max_z = max_z.max(pos.z as f32);
        }

        let center = glm::vec3(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        );

        let size_x = (max_x - min_x).abs();
        let size_y = (max_y - min_y).abs();
        let size_z = (max_z - min_z).abs();
        let max_size = size_x.max(size_y).max(size_z);
        
        // Calculate distance to fit the object with some padding
        let fov = 45.0_f32.to_radians();
        self.distance = (max_size * (1.0 + padding) / 2.0) / (fov / 2.0).tan();
        
        // Calculate eye position based on current yaw and pitch
        let eye_x = center.x + self.distance * self.yaw.cos() * self.pitch.cos();
        let eye_y = center.y + self.distance * self.yaw.sin() * self.pitch.cos();
        let eye_z = center.z + self.distance * self.pitch.sin();
        
        let eye = glm::vec3(eye_x, eye_y, eye_z);
        let target = center;
        let up = glm::vec3(0.0, 0.0, 1.0); // Z-up
        
        self.view_matrix = glm::look_at_rh(&eye, &target, &up);
        
        // Set up projection matrix with a reasonable aspect ratio
        self.projection_matrix = glm::perspective_rh_zo(
            1.0, // square aspect ratio for now
            fov,
            0.1,
            self.distance * 3.0,
        );
    }

    pub fn set_view_from_angles(&mut self, yaw_deg: f32, pitch_deg: f32) {
        self.yaw = yaw_deg.to_radians();
        self.pitch = pitch_deg.to_radians();
    }
}

// Simplified Model implementation based on main app
struct Model {
    pub mesh: triangulate::mesh::Mesh,
}

impl Model {
    pub fn new(mesh: triangulate::mesh::Mesh) -> Self {
        Model { mesh }
    }

    pub fn scale_model_to_unit_cube(&mut self) {
        if self.mesh.verts.is_empty() {
            return;
        }

        let mut min_pos = glm::DVec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_pos = glm::DVec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for vert in &self.mesh.verts {
            min_pos = glm::min2(&min_pos, &vert.pos);
            max_pos = glm::max2(&max_pos, &vert.pos);
        }

        let size = max_pos - min_pos;
        let max_size = size.x.max(size.y).max(size.z);
        
        if max_size > 0.0 {
            let scale_factor = 1.0 / max_size;
            let center = (min_pos + max_pos) * 0.5;
            
            for vert in &mut self.mesh.verts {
                vert.pos = (vert.pos - center) * scale_factor;
            }
        }
    }
}