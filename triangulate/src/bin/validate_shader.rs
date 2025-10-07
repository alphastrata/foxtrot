use naga::front::wgsl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string("./src/wgpu_triangulate/transform_mesh.wgsl")?;
    let _module = wgsl::parse_str(&source)?;
    println!("Shader validation succeeded!");
    Ok(())
}