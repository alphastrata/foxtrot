// A vertex from the CPU-generated template mesh
struct GpuTemplateVertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
    color: vec4<f32>,
};

// A transform to apply
struct GpuTransform {
    // The main transformation matrix for positions
    transform: mat4x4<f32>,
    // The inverse-transpose of the transform, for correctly transforming normals
    inverse_transpose: mat4x4<f32>,
};

// The final output vertex
struct GpuOutputVertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
    color: vec4<f32>,
};

// --- Buffers ---
@group(0) @binding(0) var<uniform> surface: GpuSurface;
@group(0) @binding(1) var<storage, read> template_verts: array<GpuTemplateVertex>;
@group(0) @binding(2) var<storage, read> transforms: array<GpuTransform>;
@group(0) @binding(3) var<storage, read_write> output_verts: array<GpuOutputVertex>;


@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;

    let num_template_verts = arrayLength(&template_verts);
    if (num_template_verts == 0u) {
        return;
    }

    // Determine which instance and which vertex in template this corresponds to
    let instance_idx = out_idx / num_template_verts;
    let template_v_idx = out_idx % num_template_verts;

    if (instance_idx >= arrayLength(&transforms)) {
        return;
    }

    // Fetch the data
    let xform = transforms[instance_idx];
    let template_v = template_verts[template_v_idx];

    // Apply transformations
    let world_pos = xform.transform * template_v.pos;
    let world_norm = normalize(xform.inverse_transpose * template_v.norm);

    // Write the result
    output_verts[out_idx] = GpuOutputVertex(
        pos: world_pos,
        norm: world_norm,
        color: template_v.color,
    );
}