// Standalone shader validation tool that doesn't depend on the main crate
// This avoids pulling in all the wgpu dependencies just for shader validation

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string("./src/wgpu_triangulate/transform_mesh.wgsl")?;

    // Use naga directly without going through the main crate
    let _module = naga::front::wgsl::parse_str(&source)?;
    println!("Shader validation succeeded!");
    Ok(())
}
