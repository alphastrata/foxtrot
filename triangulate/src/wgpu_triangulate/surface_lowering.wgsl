// Surface type constants - must match constants in wgpu_impl.rs
const SURFACE_TYPE_PLANE: u32 = 0;
const SURFACE_TYPE_CYLINDER: u32 = 1;
const SURFACE_TYPE_SPHERE: u32 = 2;
const SURFACE_TYPE_TORUS: u32 = 3;
const SURFACE_TYPE_CONE: u32 = 4;
const SURFACE_TYPE_BSPLINE: u32 = 5;
const SURFACE_TYPE_NURBS: u32 = 6;

const PI: f32 = 3.1415926535;
const EPSILON: f32 = 1.0e-6;

struct GpuSurface {
    surface_type: u32,
    radius: f32,
    major_radius: f32,
    minor_radius: f32,
    mat: mat4x4<f32>,
    mat_i: mat4x4<f32>,
    z_min: f32,
    z_max: f32,
    angle: f32,
    _p0: u32,
    location: vec3<f32>,
    _p1: f32,
    axis: vec3<f32>,
    _p2: f32,
};

// Lower 3D point to 2D UV coordinate
fn lower(p: vec3<f32>, surf: GpuSurface) -> vec2<f32> {
    let p_ = vec4<f32>(p, 1.0);
    switch surf.surface_type {
        case SURFACE_TYPE_PLANE: {
            return (surf.mat_i * p_).xy;
        }
        case SURFACE_TYPE_CONE: {
            let xy = (surf.mat_i * p_).xy;
            return vec2<f32>(-xy.x, xy.y);
        }
        case SURFACE_TYPE_CYLINDER: {
            let p_local = surf.mat_i * p_;
            // We convert the Z coordinates to either add or subtract from
            // the radius, so that we maintain the right topology
            let z = (p_local.z - surf.z_min) / (surf.z_max - surf.z_min);
            let scale = 1.0 / (1.0 + z);
            return vec2<f32>(p_local.x * scale, p_local.y * scale);
        }
        case SURFACE_TYPE_SPHERE: {
            // mat_i is constructed in prepare to be a reasonable basis
            let p_local = (surf.mat_i * p_).xyz / surf.radius;
            let r = length(p_local.yz);

            // Angle from 0 to PI
            let angle = atan2(r, p_local.x);
            let yz = p_local.yz;
            if (length(yz) < EPSILON) {
                return yz;
            } else {
                return yz * angle / length(yz);
            }
        }
        case SURFACE_TYPE_TORUS: {
            let p_local = (surf.mat_i * p_).xyz;
            let major_angle = atan2(p_local.y, p_local.z);

            // Rotate the point so that it's got Y = 0, so we can calculate
            // the minor angle
            let z = vec3<f32>(0.0, sin(major_angle), cos(major_angle));
            let z_world = vec3<f32>(z.x, surf.major_radius * z.y, surf.major_radius * z.z);
            let new_mat = make_rigid_transform(z, vec3<f32>(1.0, 0.0, 0.0), z_world);
            let new_mat_i = rigid_inverse(new_mat);
            let new_p = new_mat_i * vec4<f32>(p_local, 1.0);

            let minor_angle = atan2(new_p.x, new_p.z);

            // Construct nested circles with a scale based on the ratio
            // of radiuses (to make an _attempt_ to match 3D distance)
            let scale = 1.0 + (surf.major_radius / surf.minor_radius) * (major_angle + PI) / (2.0 * PI);

            let x = select(-cos(minor_angle), cos(minor_angle), surf.major_radius > 0.0);
            return scale * vec2<f32>(x, sin(minor_angle));
        }
        // For BSpline and NURBS, we'll need to implement additional logic or handle on CPU
        default: {
            return vec2<f32>(0.0, 0.0);
        }
    }
}

// Helper function to create a rigid transform matrix
fn make_rigid_transform(z_world: vec3<f32>, x_world: vec3<f32>, origin_world: vec3<f32>) -> mat4x4<f32> {
    var mat: mat4x4<f32>;
    mat[0] = vec4<f32>(x_world, 0.0);
    mat[1] = vec4<f32>(cross(z_world, x_world), 0.0);
    mat[2] = vec4<f32>(z_world, 0.0);
    mat[3] = vec4<f32>(origin_world, 1.0);
    return mat;
}

// Helper function for matrix inversion
fn rigid_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let r0 = m[0].xyz;
    let r1 = m[1].xyz;
    let r2 = m[2].xyz;
    let t = m[3].xyz;

    let r_inv = mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x),
        vec3<f32>(r0.y, r1.y, r2.y),
        vec3<f32>(r0.z, r1.z, r2.z)
    );

    let t_inv = -(r_inv * t); // Fixed: explicit operation with parentheses

    return mat4x4<f32>(
        vec4<f32>(r_inv[0], 0.0),
        vec4<f32>(r_inv[1], 0.0),
        vec4<f32>(r_inv[2], 0.0),
        vec4<f32>(t_inv, 1.0)
    );
}

// Input and output storage buffers for lowering
@group(0) @binding(0) var<uniform> surface: GpuSurface;
@group(0) @binding(1) var<storage, read> input_vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_vertices: array<vec4<f32>>;

// Compute shader for lowering 3D points to 2D UV coordinates
@compute 
@workgroup_size(64)
fn lowering_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= arrayLength(&input_vertices)) {
        return;
    }
    
    let pos = input_vertices[idx].xyz;
    let uv = lower(pos, surface);
    output_vertices[idx] = vec4<f32>(uv.x, uv.y, 0.0, 0.0);
}