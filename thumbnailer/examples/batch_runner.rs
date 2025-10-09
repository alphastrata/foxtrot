use clap::Parser;
use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::time::Instant;
use triangulate::wgpu_triangulate::GPUContext;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory containing STEP files
    #[arg(short, long, default_value = "./examples")]
    input_dir: String,

    /// Output directory for processed STL files
    #[arg(short, long, default_value = "./output")]
    output_dir: String,

    /// Maximum number of files to process in a single batch (max 10 recommended)
    #[arg(short, long, default_value = "5", value_parser = clap::value_parser!(u32).range(1..=10))]
    batch_size: u32,
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

    println!("Found {} STEP files in {}", step_files.len(), args.input_dir);

    // Initialize GPU context once for all processing
    let gpu_context = GPUContext::new()?;
    println!("GPU context initialized successfully");

    // Process files in batches
    let start_time = Instant::now();
    let mut total_processed = 0;

    for batch_chunk in step_files.chunks(args.batch_size as usize) {
        println!("Processing batch of {} files...", batch_chunk.len());

        let batch_start = Instant::now();

        // Process each file in the current batch using the GPU context
        for (i, file_path) in batch_chunk.iter().enumerate() {
            println!("  Processing file {}/{}: {}", i + 1, batch_chunk.len(), Path::new(file_path).file_name().unwrap().to_string_lossy());

            // Read and parse the STEP file
            let contents = fs::read(file_path)?;
            let flattened = step::step_file::StepFile::strip_flatten(&contents);
            let step_file = step::step_file::StepFile::parse(&flattened);

            // Process using GPU context
            let (mesh, _stats) = triangulate::triangulate::wgpu_triangulate_with_context(&step_file, &gpu_context);

            // Convert to STL and save
            let output_path = Path::new(&args.output_dir)
                .join(Path::new(file_path).file_stem().unwrap())
                .with_extension("stl");

            // Write the mesh as STL
            let mut stl_content = Vec::new();
            for triangle in &mesh.triangles {
                // Get the three vertices of the triangle
                let v1 = mesh.verts[triangle.verts.x as usize].pos;
                let v2 = mesh.verts[triangle.verts.y as usize].pos;
                let v3 = mesh.verts[triangle.verts.z as usize].pos;

                // Calculate normal using cross product
                let edge1 = nalgebra_glm::DVec3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
                let edge2 = nalgebra_glm::DVec3::new(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);
                let normal = nalgebra_glm::cross(&edge1, &edge2);
                let normal = normal.normalize();

                // Add to STL content as binary format (simplified)
                // Write normal (3 floats) + 3 vertices (9 floats) + 1 uint16 (attribute byte count)
                stl_content.extend_from_slice(&(normal.x as f32).to_le_bytes());
                stl_content.extend_from_slice(&(normal.y as f32).to_le_bytes());
                stl_content.extend_from_slice(&(normal.z as f32).to_le_bytes());

                // Vertex 1
                stl_content.extend_from_slice(&(v1.x as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v1.y as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v1.z as f32).to_le_bytes());

                // Vertex 2
                stl_content.extend_from_slice(&(v2.x as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v2.y as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v2.z as f32).to_le_bytes());

                // Vertex 3
                stl_content.extend_from_slice(&(v3.x as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v3.y as f32).to_le_bytes());
                stl_content.extend_from_slice(&(v3.z as f32).to_le_bytes());

                // Attribute byte count
                stl_content.extend_from_slice(&[0u8, 0u8]);
            }

            // Write the STL to file with proper binary STL header
            let mut output_file = std::fs::File::create(&output_path)?;
            use std::io::Write;
            
            // Write STL header (80 bytes)
            let header = format!("Generated by batch_runner - {}", output_path.file_name().unwrap().to_string_lossy());
            let mut full_header = [0u8; 80];
            let header_bytes = header.as_bytes();
            let copy_len = std::cmp::min(header_bytes.len(), 80);
            full_header[..copy_len].copy_from_slice(&header_bytes[..copy_len]);
            output_file.write_all(&full_header)?;
            
            // Write number of triangles
            let num_triangles = (stl_content.len() / 50) as u32; // 50 bytes per triangle in binary STL
            output_file.write_all(&num_triangles.to_le_bytes())?;
            
            // Write triangle data
            output_file.write_all(&stl_content)?;

            println!("    Saved to: {}", output_path.display());
        }

        let batch_duration = batch_start.elapsed();
        println!("  Batch completed in {:.2?}", batch_duration);
        total_processed += batch_chunk.len();
    }

    let total_duration = start_time.elapsed();
    println!("\nProcessed {} files in {:.2?}", total_processed, total_duration);
    println!("Average processing time per file: {:.2?}", total_duration / total_processed.max(1) as u32);

    Ok(())
}