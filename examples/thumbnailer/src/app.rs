// ... (at the top of the file)
use nalgebra_glm as glm;
use std::time::SystemTime;
use triangulate::mesh::Mesh;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

// --- ADD: A simple struct for our line color uniform ---
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LineStyle {
    color: [f32; 4],
}

pub struct App {
    pub start_time: SystemTime,
    pub window_size: PhysicalSize<u32>,
    pub graphics_adapter: wgpu::Adapter,
    pub graphics_device: wgpu::Device,
    pub command_queue: wgpu::Queue,
}

impl App {
    pub fn new(
        start_time: SystemTime,
        window_size: PhysicalSize<u32>,
        graphics_adapter: wgpu::Adapter,
        _surface: Option<wgpu::Surface>,
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
        args: &super::Args,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if mode_to_string(&args.mode) == "composite" {
            self.render_composite_view(texture_view, mesh, args)?;
        } else {
            self.render_single_isometric_view(texture_view, mesh, args)?;
        }

        Ok(())
    }

    fn render_single_isometric_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        mesh: &Mesh,
        args: &super::Args,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{camera::Camera, model::Model};

        // Create camera and model (same as before)
        let mut camera = Camera::new();
        let mut model = Model::new(mesh.clone());
        model.scale_model_to_unit_cube();

        let model_center = glm::vec3(0.0, 0.0, 0.0); // After scaling, center is at origin
        
        camera.fit_to_bounds(&model.mesh.verts, 1.0);
        camera.set_view(&args.view, &model_center);

        // Create Depth Texture (same as before)
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

        // --- SETUP FOR LINE RENDERING ---
        let line_shader = self.graphics_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("line_shader.wgsl"))),
        });

        let line_bind_group_layout = self.graphics_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Line Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { // Camera
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // LineStyle
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        
        let line_pipeline_layout = self.graphics_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[&line_bind_group_layout],
            push_constant_ranges: &[],
        });

        // --- PIPELINE 1: Depth Pre-pass (Solid fill) ---
        let depth_pipeline = self.graphics_device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Pre-pass Pipeline"),
            layout: Some(&line_pipeline_layout), // Can reuse layout, just won't bind the color
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 6 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: None, // No fragment shader needed for depth-only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // --- PIPELINE 2: Visible Lines ---
        let visible_lines_pipeline = self.graphics_device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Visible Lines Pipeline"),
            layout: Some(&line_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 6 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: if args.transparent { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Bgra8UnormSrgb },
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Line,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual, // Draw if in front or at the same depth
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // --- PIPELINE 3: Hidden Lines ---
        let hidden_lines_pipeline = self.graphics_device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Hidden Lines Pipeline"),
            layout: Some(&line_pipeline_layout),
            // ... Vertex State is identical to visible_lines_pipeline ...
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 6 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: if args.transparent { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Bgra8UnormSrgb },
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Blend with background
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                polygon_mode: wgpu::PolygonMode::Line,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater, // Draw only if BEHIND what's in the depth buffer
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // --- Create Vertex and Index Buffers (same as before) ---
        let vertex_data: Vec<f32> = model.mesh.verts.iter().flat_map(|v| [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32, v.norm.x as f32, v.norm.y as f32, v.norm.z as f32]).collect();
        let vertex_buffer = self.graphics_device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Vertex Buffer"), contents: bytemuck::cast_slice(&vertex_data), usage: wgpu::BufferUsages::VERTEX });
        let indices: Vec<u32> = model.mesh.triangles.iter().flat_map(|tri| [tri.verts.x, tri.verts.y, tri.verts.z]).collect();
        let index_buffer = self.graphics_device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Index Buffer"), contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX });

        // --- Create Uniform Buffers and Bind Groups ---
        let view_proj_matrix = camera.projection_matrix * camera.view_matrix;
        let camera_uniform_buffer = self.graphics_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(view_proj_matrix.as_slice()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Visible lines are black
        let visible_line_style = LineStyle { color: [0.0, 0.0, 0.0, 1.0] };
        let visible_line_buffer = self.graphics_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Visible Line Uniform"),
            contents: bytemuck::cast_slice(&[visible_line_style]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Hidden lines are a semi-transparent gray
        let hidden_line_style = LineStyle { color: [0.5, 0.5, 0.5, 0.5] };
        let hidden_line_buffer = self.graphics_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hidden Line Uniform"),
            contents: bytemuck::cast_slice(&[hidden_line_style]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let camera_only_bind_group = self.graphics_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Only Bind Group"),
            layout: &line_bind_group_layout, // Close enough, we just won't use binding 1
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: visible_line_buffer.as_entire_binding() }], // Dummy binding
        });

        let visible_lines_bind_group = self.graphics_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Visible Lines Bind Group"),
            layout: &line_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: visible_line_buffer.as_entire_binding() }],
        });

        let hidden_lines_bind_group = self.graphics_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hidden Lines Bind Group"),
            layout: &line_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: hidden_line_buffer.as_entire_binding() }],
        });

        // MSAA framebuffer (same as before)
        let multisampled_framebuffer = self.graphics_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Multisampled Framebuffer"),
            size: wgpu::Extent3d { width: args.thumbnail_size, height: args.thumbnail_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: if args.transparent { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Bgra8UnormSrgb },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_framebuffer.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.graphics_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Thumbnail Command Encoder") });

        // --- RENDER PASS 1: Depth Pre-pass ---
        {
            let mut depth_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Pre-pass"),
                color_attachments: &[], // No color attachment
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            depth_pass.set_pipeline(&depth_pipeline);
            depth_pass.set_bind_group(0, &camera_only_bind_group, &[]);
            depth_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            depth_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            depth_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
        
        // --- RENDER PASS 2: Visible and Hidden Lines ---
        {
            let mut line_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Line Render Pass"),
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
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), // Load the depth buffer from the first pass
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw Hidden Lines FIRST
            line_pass.set_pipeline(&hidden_lines_pipeline);
            line_pass.set_bind_group(0, &hidden_lines_bind_group, &[]);
            line_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            line_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            line_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);

            // Draw Visible Lines on top
            line_pass.set_pipeline(&visible_lines_pipeline);
            line_pass.set_bind_group(0, &visible_lines_bind_group, &[]);
            line_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            line_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            line_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.command_queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn render_composite_view(
        &mut self,
        texture_view: &wgpu::TextureView,
        mesh: &Mesh,
        args: &super::Args,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::{camera::Camera, model::Model};

        let mut camera = Camera::new();
        let mut model = Model::new(mesh.clone());

        model.scale_model_to_unit_cube();

        // Use the transformed vertices from the `model` to fit the camera
        camera.fit_to_bounds(&model.mesh.verts, 1.0);

        // Set isometric view angles while keeping the distance calculated by fit_to_bounds
        camera.set_view_from_angles(45.0, 35.0); // 45° azimuth, 35° elevation for isometric view

        // STEP 1: Create Depth Texture
        let depth_texture_size = wgpu::Extent3d {
            width: args.thumbnail_size,
            height: args.thumbnail_size,
            depth_or_array_layers: 1,
        };
        let depth_texture = self
            .graphics_device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: depth_texture_size,
                mip_level_count: 1,
                sample_count: 4, // A common value for MSAA. 2, 4, or 8 are typical.
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float, // A common depth format
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Load shader
        let shader = self
            .graphics_device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Thumbnail Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "shader.wgsl"
                ))),
            });

        // Create bind group layout
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
                            visibility: wgpu::ShaderStages::FRAGMENT, // Light is used in fragment shader
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

        // Create render pipeline
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
                        depth_compare: wgpu::CompareFunction::Less, // Standard depth test
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 4,                         // A common value for MSAA. 2, 4, or 8 are typical.
                        mask: !0,                         // Use all samples
                        alpha_to_coverage_enabled: false, // No transparency effects needed
                    },
                    multiview: None,
                    cache: None,
                });

        // Use the transformed mesh from the `model` to create the vertex buffer
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

        // Use the transformed mesh for indices as well
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

        // Create uniform buffer for camera matrices (using the main camera for now)
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
            // Light coming from isometric direction (45°, 35°) for consistent lighting
            direction: [0.5, 0.5, 0.7], // normalized in shader
            // Make the light brighter and more white
            color: [1.0, 1.0, 1.0],
            _pad: [0.0, 0.0], // padding to align to 16-byte boundary
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

        // --- MSAA STEP 4: Create a multisampled texture to render into ---
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

        let mut encoder =
            self.graphics_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Thumbnail Command Encoder"),
                });

        // Create the render pass
        {
            // --- MSAA STEP 5: Update the render pass to use the multisampled view and resolve target ---
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Thumbnail Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Render TO the multisampled view
                    view: &multisampled_view,
                    // Resolve (downsample) INTO the final texture_view
                    resolve_target: Some(texture_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if args.transparent {
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
                })],
                // The depth attachment must also be multisampled
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear to the farthest distance
                        store: wgpu::StoreOp::Store,
                    }),
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

        // Submit the command buffer
        self.command_queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    pub async fn read_texture_to_buffer(
        &self,
        texture: &wgpu::Texture,
        texture_size: &wgpu::Extent3d,
    ) -> Vec<u8> {
        let device = &self.graphics_device;

        // Calculate aligned bytes per row
        let bytes_per_row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = texture_size.width * 4;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(bytes_per_row_alignment) * bytes_per_row_alignment;

        // Create a buffer to copy the texture to (with padding for alignment)
        let buffer_size = (padded_bytes_per_row * texture_size.height) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Thumbnail Read Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Thumbnail Read Encoder"),
        });

        // Calculate aligned bytes per row
        let bytes_per_row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = texture_size.width * 4;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(bytes_per_row_alignment) * bytes_per_row_alignment;

        // Copy the texture to the buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(texture_size.height),
                },
            },
            *texture_size,
        );

        self.command_queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read the data
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if tx.send(result).is_err() {
                eprintln!("Failed to send mapping result");
            }
        });

        // Wait for the mapping to complete
        // Since this is a blocking operation in an async context, use device.poll to wait
        _= self.graphics_device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        match rx.recv() {
            Ok(Ok(())) => {
                let mapped_buffer = buffer_slice.get_mapped_range();

                // Calculate the aligned bytes per row again
                let bytes_per_row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let unpadded_bytes_per_row = (texture_size.width * 4) as usize;
                let padded_bytes_per_row = unpadded_bytes_per_row
                    .div_ceil(bytes_per_row_alignment as usize)
                    * bytes_per_row_alignment as usize;

                // Extract the unpadded data row by row
                let mut data =
                    Vec::with_capacity((texture_size.width * texture_size.height * 4) as usize);
                for row in 0..texture_size.height as usize {
                    let start_idx = row * padded_bytes_per_row;
                    let end_idx = start_idx + unpadded_bytes_per_row;
                    data.extend_from_slice(&mapped_buffer[start_idx..end_idx]);
                }

                drop(mapped_buffer);
                buffer.unmap();
                data
            }
            _ => {
                eprintln!("Failed to map buffer for reading");
                vec![255; (texture_size.width * texture_size.height * 4) as usize] // Return white pixels as fallback
            }
        }
    }
}

use crate::cli::RenderMode;

fn mode_to_string(mode: &RenderMode) -> String {
    match mode {
        RenderMode::Iso => "iso".to_string(),
        RenderMode::Composite => "composite".to_string(),
    }
}
