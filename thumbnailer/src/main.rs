use clap::Parser;
use std::io::Write;
use std::time::SystemTime;
use tracing::trace;

mod app;
mod camera;
mod cli;
mod img_utils;
mod model;
mod wgpu_utils;

use cli::{Args, TriangulationEngine};

#[cfg(feature = "occt")]
use opencascade::primitives::Shape;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let start_time = std::time::Instant::now();

    let args = Args::parse();
    trace!("Starting thumbnail generation for: {}", args.input);

    let engine = args.engine.clone();
    let thumbnail_size = args.thumbnail_size;
    let transparent = args.transparent;
    let decimation_ratio = args.decimation_ratio;
    let decimation_error = args.decimation_error;

    #[cfg(feature = "occt")]
    let occt_linear_deflection = args.occt_linear_deflection;
    #[cfg(feature = "occt")]
    let occt_angular_deflection = args.occt_angular_deflection;

    let input = args.input.clone();
    let engine_for_thread = engine.clone();
    let loader = std::thread::spawn(move || {
        let foxtrot_pipeline_start = std::time::Instant::now();
        use step::step_file::StepFile;
        use triangulate::triangulate::wgpu_triangulate as triangulate;

        let data = std::fs::read(&input).expect("Could not open file");
        let flat = StepFile::strip_flatten(&data);
        let step = StepFile::parse(&flat);
        trace!("STEP file parsed in {:?}", foxtrot_pipeline_start.elapsed());

        match engine_for_thread {
            TriangulationEngine::Foxtrot => {
                let (mut mesh, _stats) = triangulate(&step);
                trace!(
                    "Mesh triangulation in {:?}, starting with {} vertices and {} triangles.",
                    foxtrot_pipeline_start.elapsed(),
                    mesh.verts.len(),
                    mesh.triangles.len()
                );

                let decimation_start = std::time::Instant::now();
                let original_vertex_count = mesh.verts.len();
                let original_triangle_count = mesh.triangles.len();

                trace!(
                    "Starting mesh decimation from {} vertices and {} triangles",
                    original_vertex_count, original_triangle_count
                );

                let vertices: Vec<f32> = mesh
                    .verts
                    .iter()
                    .flat_map(|v| [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32])
                    .collect();

                let indices: Vec<u32> = mesh
                    .triangles
                    .iter()
                    .flat_map(|t| [t.verts.x, t.verts.y, t.verts.z])
                    .collect();

                let target_index_count = (indices.len() as f32 * decimation_ratio) as usize;
                let target_error = decimation_error;

                let vertex_adapter = meshopt::VertexDataAdapter::new(
                    bytemuck::cast_slice(&vertices),
                    3 * std::mem::size_of::<f32>(),
                    0,
                )
                .unwrap();

                let mut error_result: f32 = 0.0;
                let simplified_indices = meshopt::simplify(
                    &indices,
                    &vertex_adapter,
                    target_index_count,
                    target_error,
                    meshopt::SimplifyOptions::LockBorder,
                    Some(&mut error_result),
                );

                let new_triangles = simplified_indices
                    .chunks_exact(3)
                    .map(|chunk| triangulate::mesh::Triangle {
                        verts: nalgebra_glm::U32Vec3::new(chunk[0], chunk[1], chunk[2]),
                    })
                    .collect();

                mesh.triangles = new_triangles;

                trace!(
                    "Mesh decimation completed in {:?}. Decimated to {} vertices and {} triangles.",
                    decimation_start.elapsed(),
                    mesh.verts.len(),
                    mesh.triangles.len()
                );

                mesh
            }
            #[cfg(feature = "occt")]
            TriangulationEngine::Occt => {
                let occt_pipeline_start = std::time::Instant::now();
                trace!(
                    "Using OCCT engine with linear_deflection={} angular_deflection={}",
                    occt_linear_deflection, occt_angular_deflection
                );

                let shape_to_mesh = Shape::read_step(input).expect("OCCT failed to read STEP file");

                use opencascade::mesh::Mesher;
                let occt_mesh = Mesher::new(&shape_to_mesh).mesh();

                trace!(
                    "OCCT triangulation and mesh unification completed in {:?}",
                    occt_pipeline_start.elapsed()
                );

                let mut final_mesh = triangulate::mesh::Mesh {
                    verts: occt_mesh
                        .vertices
                        .iter()
                        .zip(occt_mesh.normals.iter())
                        .map(|(v, n)| triangulate::mesh::Vertex {
                            pos: nalgebra_glm::DVec3::new(v.x, v.y, v.z),
                            norm: nalgebra_glm::DVec3::new(n.x, n.y, n.z),
                            color: nalgebra_glm::DVec3::new(0.5, 0.5, 0.5),
                        })
                        .collect(),
                    triangles: occt_mesh
                        .indices
                        .chunks_exact(3)
                        .map(|tri| triangulate::mesh::Triangle {
                            verts: nalgebra_glm::U32Vec3::new(
                                tri[0] as u32,
                                tri[1] as u32,
                                tri[2] as u32,
                            ),
                        })
                        .collect(),
                };

                trace!(
                    "Converted to local mesh with {} vertices and {} triangles.",
                    final_mesh.verts.len(),
                    final_mesh.triangles.len()
                );

                let decimation_start = std::time::Instant::now();
                let original_vertex_count = final_mesh.verts.len();
                let original_triangle_count = final_mesh.triangles.len();

                trace!(
                    "Starting mesh decimation from {} vertices and {} triangles",
                    original_vertex_count, original_triangle_count
                );

                let vertices: Vec<f32> = final_mesh
                    .verts
                    .iter()
                    .flat_map(|v| [v.pos.x as f32, v.pos.y as f32, v.pos.z as f32])
                    .collect();

                let indices: Vec<u32> = final_mesh
                    .triangles
                    .iter()
                    .flat_map(|t| [t.verts.x, t.verts.y, t.verts.z])
                    .collect();

                let target_index_count = (indices.len() as f32 * decimation_ratio) as usize;
                let target_error = decimation_error;

                let vertex_adapter = meshopt::VertexDataAdapter::new(
                    bytemuck::cast_slice(&vertices),
                    3 * std::mem::size_of::<f32>(),
                    0,
                )
                .unwrap();

                let mut error_result: f32 = 0.0;
                let simplified_indices = meshopt::simplify(
                    &indices,
                    &vertex_adapter,
                    target_index_count,
                    target_error,
                    meshopt::SimplifyOptions::LockBorder,
                    Some(&mut error_result),
                );

                let new_triangles = simplified_indices
                    .chunks_exact(3)
                    .map(|chunk| triangulate::mesh::Triangle {
                        verts: nalgebra_glm::U32Vec3::new(chunk[0], chunk[1], chunk[2]),
                    })
                    .collect();

                final_mesh.triangles = new_triangles;

                trace!(
                    "Mesh decimation completed in {:?}. Decimated to {} vertices and {} triangles.",
                    decimation_start.elapsed(),
                    final_mesh.verts.len(),
                    final_mesh.triangles.len()
                );

                final_mesh
            }
        }
    });

    let (device, queue) = wgpu_utils::create_wgpu_device()?;

    let mesh = loader.join().expect("Loader thread failed");

    trace!(
        "Generated thumbnail with {} vertices and {} triangles after decimation",
        mesh.verts.len(),
        mesh.triangles.len()
    );

    if mesh.verts.is_empty() || mesh.triangles.is_empty() {
        eprintln!(
            "Warning: Mesh is empty ({} vertices, {} triangles). Creating empty PNG with message.",
            mesh.verts.len(),
            mesh.triangles.len()
        );

        let mut png_data =
            Vec::<u8>::with_capacity((args.thumbnail_size * args.thumbnail_size * 4) as usize);

        let bg_color = if args.transparent {
            vec![0, 0, 0, 0]
        } else {
            vec![255, 255, 255, 255]
        };

        (0..(args.thumbnail_size * args.thumbnail_size)).for_each(|_| {
            png_data.extend_from_slice(&bg_color);
        });

        (0..args.thumbnail_size).for_each(|i| {
            let idx = ((i * args.thumbnail_size + i) * 4) as usize;
            if idx < png_data.len() - 3 {
                png_data[idx] = 255;
                png_data[idx + 1] = 0;
                png_data[idx + 2] = 0;
                if !args.transparent {
                    png_data[idx + 3] = 255;
                }
            }
        });

        let mut output_png_data = Vec::<u8>::with_capacity(png_data.len());
        let mut encoder = png::Encoder::new(
            std::io::Cursor::new(&mut output_png_data),
            args.thumbnail_size,
            args.thumbnail_size,
        );
        encoder.set_color(png::ColorType::Rgba);

        let mut png_writer = encoder.write_header()?;
        png_writer.write_image_data(&png_data[..])?;
        png_writer.finish()?;

        let mut file = std::fs::File::create(&args.output)?;
        file.write_all(&output_png_data)?;

        return Ok(());
    }

    let (texture, texture_view) = wgpu_utils::create_render_texture(
        &device,
        args.thumbnail_size,
        args.thumbnail_size,
        args.transparent,
    );

    let mut app = app::App::new(
        SystemTime::now(),
        winit::dpi::PhysicalSize::new(thumbnail_size, thumbnail_size),
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            }))
            .expect("Failed to find an appropriate adapter")
        },
        None,
        device,
        queue,
    );

    let render_start = std::time::Instant::now();
    let render_args = args.clone();
    app.render_to_texture(&texture_view, &mesh, &render_args)?;
    trace!("Render completed in {:?}", render_start.elapsed());

    let buffer = pollster::block_on(app.read_texture_to_buffer(
        &texture,
        &wgpu::Extent3d {
            width: args.thumbnail_size,
            height: args.thumbnail_size,
            depth_or_array_layers: 1,
        },
    ));
    let img_data = if transparent {
        buffer
    } else {
        let mut rgba_buffer = Vec::with_capacity((thumbnail_size * thumbnail_size * 4) as usize);
        for chunk in buffer.chunks(4) {
            rgba_buffer.push(chunk[2]);
            rgba_buffer.push(chunk[1]);
            rgba_buffer.push(chunk[0]);
            rgba_buffer.push(chunk[3]);
        }
        rgba_buffer
    };

    assert!(
        !img_data.iter().all(|&b| b == 255),
        "Image data contains ALL 255 values, this is a bug fix the code that makes the thumbnail, do NOT remove this check."
    );

    let mut flipped_img_data = Vec::with_capacity(img_data.len());
    let row_size = (thumbnail_size * 4) as usize;

    for row in (0..thumbnail_size).rev() {
        let start_idx = row as usize * thumbnail_size as usize * 4;
        let end_idx = start_idx + row_size;
        flipped_img_data.extend_from_slice(&img_data[start_idx..end_idx]);
    }

    let cropped_img_data = if args.crop {
        img_utils::auto_crop_image(
            &flipped_img_data,
            args.thumbnail_size,
            args.thumbnail_size,
            args.transparent,
        )
    } else {
        flipped_img_data
    };

    let mut png_data = Vec::<u8>::with_capacity(cropped_img_data.len());
    let mut encoder = png::Encoder::new(
        std::io::Cursor::new(&mut png_data),
        thumbnail_size,
        thumbnail_size,
    );
    encoder.set_color(png::ColorType::Rgba);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(&cropped_img_data[..])?;
    png_writer.finish()?;

    let mut file = std::fs::File::create(&args.output)?;
    file.write_all(&png_data)?;

    trace!("Thumbnail saved to {}", args.output);
    trace!("Walltime: {:?}ms", start_time.elapsed().as_millis());

    Ok(())
}
