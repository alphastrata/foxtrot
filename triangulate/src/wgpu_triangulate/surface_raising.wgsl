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

struct GpuTransform {
    transform: mat4x4<f32>,
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

// Calculate surface normal
fn normal(p: vec3<f32>, uv: vec2<f32>, surf: GpuSurface) -> vec3<f32> {
    switch surf.surface_type {
        case SURFACE_TYPE_PLANE: {
            return surf.axis; // normal is stored in axis
        }
        case SURFACE_TYPE_CONE: {
            // Project into CONE SPACE
            let pos = surf.mat_i * vec4<f32>(p, 1.0);
            var xy = normalize(pos.xy);
            if (length(pos.xy) < EPSILON) {
                return vec3<f32>(0.0, 0.0, 0.0);
            }
            let normal = vec4<f32>(xy.x * cos(surf.angle), xy.y * cos(surf.angle), -sin(surf.angle), 0.0);
            // Deproject back into world space
            return normalize((surf.mat * normal).xyz);
        }
        case SURFACE_TYPE_SPHERE: {
            return normalize(p - surf.location);
        }
        case SURFACE_TYPE_CYLINDER: {
            // Project the point onto the axis
            let proj = surf.mat_i * vec4<f32>(p, 1.0);
            // Then the normal is just pointing along that direction
            let norm = normalize(vec3<f32>(proj.x, proj.y, 0.0));
            return normalize((surf.mat * vec4<f32>(norm, 0.0)).xyz);
        }
        case SURFACE_TYPE_TORUS: {
            let p_local = (surf.mat_i * vec4<f32>(p, 1.0)).xyz;
            let major_angle = atan2(p_local.y, p_local.z);

            let z = vec3<f32>(0.0, sin(major_angle), cos(major_angle)) * surf.major_radius;
            let norm = normalize(p_local - z);

            return normalize((surf.mat * vec4<f32>(norm, 0.0)).xyz);
        }
        // For BSpline and NURBS, we'll need to implement additional logic
        default: {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
    }
}

// Raise 2D UV coordinate back to 3D point
fn raise(uv: vec2<f32>, surf: GpuSurface) -> vec3<f32> {
    switch surf.surface_type {
        case SURFACE_TYPE_PLANE: {
            return (surf.mat * vec4<f32>(uv, 0.0, 1.0)).xyz;
        }
        case SURFACE_TYPE_SPHERE: {
            let angle = length(uv);
            if (angle > PI) {
                return vec3<f32>(0.0, 0.0, 0.0);
            }
            let x = cos(angle);
            var pos: vec3<f32>;
            if (length(uv) < EPSILON) {
                pos = vec3<f32>(x, 0.0, 0.0);
            } else {
                let yz = normalize(uv) * sin(angle);
                pos = vec3<f32>(x, yz.x, yz.y);
            }
            pos = pos * surf.radius;
            return (surf.mat * vec4<f32>(pos, 1.0)).xyz;
        }
        // For BSpline and NURBS, we'll implement more complex logic or handle on CPU
        default: {
            return vec3<f32>(0.0, 0.0, 0.0);
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

// Input and output storage buffers for raising and transforms
@group(0) @binding(0) var<uniform> surface: GpuSurface;
@group(0) @binding(1) var<storage, read> template_uvs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> template_normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> transforms: array<GpuTransform>;
@group(0) @binding(4) var<storage, read_write> output_positions: array<vec4<f32>>;

// Compute shader for raising 2D points to 3D and applying transformations
@compute 
@workgroup_size(64)
fn raising_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    // Calculate which instance and which vertex in template this corresponds to
    let num_template_verts = arrayLength(&template_uvs);
    if (num_template_verts == 0) {
        return;
    }
    
    let instance_idx = idx / num_template_verts;
    let vertex_in_template = idx % num_template_verts;
    
    // Bounds check
    if (instance_idx >= arrayLength(&transforms)) {
        return;
    }
    
    let template_uv = template_uvs[vertex_in_template].xy;
    let template_normal = template_normals[vertex_in_template].xyz;
    
    // Raise from 2D UV to 3D local space
    let local_pos = raise(template_uv, surface);
    
    // Apply the instance transform
    let world_pos = (transforms[instance_idx].transform * vec4<f32>(local_pos, 1.0)).xyz;
    let world_normal = normalize((transforms[instance_idx].transform * vec4<f32>(template_normal, 0.0)).xyz); // Normals use 0.0 for w
    
    output_positions[idx] = vec4<f32>(world_pos, 1.0);
}