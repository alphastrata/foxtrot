use clap::Parser;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// Add necessary imports for rendering
use nalgebra_glm as glm;
use pollster;
use wgpu;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// Add PNG encoding
use png;

// Define simple structures needed for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

struct Camera {
    eye: glm::DVec3,
    target: glm::DVec3,
    up: glm::DVec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn new(
        pos: [f64; 3],
        _yaw: f64,
        _pitch: f64,
        fovy: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let eye = glm::DVec3::new(pos[0], pos[1], pos[2]);
        let target = glm::DVec3::new(0.0, 0.0, 0.0);
        let up = glm::DVec3::new(0.0, 1.0, 0.0);

        Camera {
            eye,
            target,
            up,
            aspect,
            fovy,
            znear,
            zfar,
        }
    }

    fn build_view_projection_matrix(&self) -> glm::Mat4 {
        let view = glm::look_at_rh(
            &glm::vec3(self.eye.x as f32, self.eye.y as f32, self.eye.z as f32),
            &glm::vec3(
                self.target.x as f32,
                self.target.y as f32,
                self.target.z as f32,
            ),
            &glm::vec3(self.up.x as f32, self.up.y as f32, self.up.z as f32),
        );
        let proj = glm::perspective_rh_zo(self.aspect, self.fovy, self.znear, self.zfar);
        proj * view
    }
}

// Simple renderer for batch processing
struct BatchRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
}

impl BatchRenderer {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batch Render Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../src/shader.wgsl"
            ))),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("uniform_bind_group_layout"),
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Batch Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Batch Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            shader,
            uniform_bind_group_layout,
            render_pipeline_layout,
            render_pipeline,
        }
    }

    fn render_mesh_to_png(
        &self,
        mesh: &triangulate::mesh::Mesh,
        output_path: &str,
        thumbnail_size: Option<u32>,
        transparent: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let size = PhysicalSize::new(thumbnail_size.unwrap_or(512), thumbnail_size.unwrap_or(512));

        // Create texture to render to
        let texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: Some("Batch Render Texture"),
            view_formats: &[],
        };
        let texture = self.device.create_texture(&texture_descriptor);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Batch Depth Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create vertex and index buffers
        let vertices: Vec<Vertex> = mesh
            .verts
            .iter()
            .map(|v| Vertex {
                position: [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32],
                normal: [v.norm.x as f32, v.norm.y as f32, v.norm.z as f32],
            })
            .collect();

        let indices: Vec<u32> = mesh
            .triangles
            .iter()
            .flat_map(|t| [t.verts.x, t.verts.y, t.verts.z])
            .collect();

        let vertex_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Create uniform buffer for camera/view matrices
        let camera = Camera::new(
            [0.0, 0.0, 5.0],
            0.0,
            0.0,
            45.0_f32.to_radians(),
            size.width as f32 / size.height as f32,
            0.1,
            100.0,
        );
        let view_proj = camera.build_view_projection_matrix();

        // Convert the 4x4 matrix to an array of f32 values for the GPU
        let view_proj_array: [f32; 16] = [
            view_proj[(0, 0)] as f32,
            view_proj[(0, 1)] as f32,
            view_proj[(0, 2)] as f32,
            view_proj[(0, 3)] as f32,
            view_proj[(1, 0)] as f32,
            view_proj[(1, 1)] as f32,
            view_proj[(1, 2)] as f32,
            view_proj[(1, 3)] as f32,
            view_proj[(2, 0)] as f32,
            view_proj[(2, 1)] as f32,
            view_proj[(2, 2)] as f32,
            view_proj[(2, 3)] as f32,
            view_proj[(3, 0)] as f32,
            view_proj[(3, 1)] as f32,
            view_proj[(3, 2)] as f32,
            view_proj[(3, 3)] as f32,
        ];

        let camera_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&view_proj_array),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create light uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Light {
            direction: [f32; 3],
            _padding1: f32, // padding to align to 16-byte boundary
            color: [f32; 3],
            _padding2: f32, // padding to align to 16-byte boundary
        }

        let light_data = Light {
            direction: [-1.0, -1.0, -1.0], // Light direction
            _padding1: 0.0,
            color: [1.0, 1.0, 1.0], // White light
            _padding2: 0.0,
        };

        let light_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batch Light Uniform Buffer"),
                contents: bytemuck::cast_slice(&[light_data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buf.as_entire_binding(),
                },
            ],
            label: Some("batch_uniform_bind_group"),
        });

        // Render to texture
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Batch Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if transparent {
                            wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }
                        } else {
                            wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buf.slice(..));
            render_pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read texture data back
        let bytes_per_row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = size.width * 4;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(bytes_per_row_alignment) * bytes_per_row_alignment;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_bytes_per_row * size.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and get the data
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();

        let img_data = buffer.slice(..).get_mapped_range().to_vec();
        drop(buffer);

        // Flip the image vertically for correct PNG orientation
        let mut flipped_img_data = Vec::with_capacity(img_data.len());
        let row_size = (size.width * 4) as usize;

        for row in (0..size.height).rev() {
            let start_idx = (row * size.width * 4) as usize;
            let end_idx = start_idx + row_size;
            if start_idx < img_data.len() {
                let end_idx = std::cmp::min(end_idx, img_data.len());
                flipped_img_data.extend_from_slice(&img_data[start_idx..end_idx]);
            }
        }

        // Encode to PNG
        let mut png_data = Vec::<u8>::with_capacity(flipped_img_data.len());
        assert!(
            !png_data.iter().all(|px| *px == 255),
            "Produced PNG is pure white, produced by a bug in the rendering."
        );

        let mut encoder =
            png::Encoder::new(std::io::Cursor::new(&mut png_data), size.width, size.height);
        encoder.set_color(png::ColorType::Rgba);
        let mut png_writer = encoder.write_header().unwrap();
        png_writer.write_image_data(&flipped_img_data[..]).unwrap();
        png_writer.finish().unwrap();

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&png_data)?;

        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

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

    // Initialize the batch renderer
    let renderer = pollster::block_on(BatchRenderer::new());
    println!("Batch renderer initialized successfully");

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

        assert!(!mesh.verts.is_empty());
        assert!(!mesh.triangles.is_empty());

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

        // Render the mesh to PNG using the batch renderer
        renderer.render_mesh_to_png(
            &mesh,
            &output_path.to_string_lossy(),
            Some(args.thumbnail_size),
            args.transparent,
        )?;

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
