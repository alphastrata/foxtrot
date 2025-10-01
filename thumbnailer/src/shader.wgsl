struct Camera {
    view_proj: mat4x4<f32>,
};

// Add the light struct
struct Light {
    direction: vec3<f32>,
    color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

// Add binding for the light uniform
@group(0) @binding(1)
var<uniform> light: Light;

// Vertex shader
@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = camera.view_proj * vec4<f32>(position, 1.0);
    output.normal = normalize(normal); // Ensure normal is normalized
    return output;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
     // Simple Lambertian 
    let light_dir = normalize(light.direction);
    let diffuse_strength = max(dot(in.normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    // Gotta colour the model something...
    let object_color = vec3<f32>(0.106, 0.353, 0.941); 

    // Increase ambient light to soften shadows
    let ambient_strength = 0.28;
    let ambient_color = object_color * ambient_strength;
    let final_color = object_color * diffuse_color + ambient_color;

    return vec4<f32>(final_color, 1.0);
}