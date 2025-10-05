# Detailed Performance Optimization Plan for GPU Triangulation

Your GPU implementation is **178x slower** than the CPU version. This is catastrophically bad and indicates fundamental architectural problems, not just minor inefficiencies.

## Root Cause Analysis

### Critical Issues

1. **CPU-GPU Transfer Overhead**: You're calling `gpu_lower_vertices` and `gpu_raise_and_transform` separately for EACH FACE, with full round-trips:
   - CPU → GPU transfer
   - GPU computation  
   - GPU → CPU transfer
   - Repeat for next face

2. **Tiny Batch Sizes**: Most faces have 50-100 vertices. Launching GPU kernels for such small workloads is worse than useless - the kernel launch overhead (~10-50μs) dominates actual compute time (~1μs).

3. **Synchronous Execution**: Every face blocks waiting for GPU completion before moving to the next one.

4. **GPU Starvation**: Your TITAN RTX has 4608 CUDA cores sitting idle 99% of the time waiting for tiny transfers.

## Performance Targets

- CPU baseline: ~50ms (from your benchmarks)
- GPU should achieve: **5-15ms** (3-10x speedup, not 178x slower)
- Currently achieving: ~900ms

## Optimization Strategy

### Phase 1: Batch All Operations (Expected: 50-100x speedup)

**Current Pattern** (per face):
```
Face 1: CPU→GPU → compute → GPU→CPU
Face 2: CPU→GPU → compute → GPU→CPU
...
Face N: CPU→GPU → compute → GPU→CPU
```

**Target Pattern** (all faces):
```
All faces: CPU→GPU → compute all → GPU→CPU
```

**Implementation**:
```rust
pub fn gpu_triangulate_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face_tasks: &[FaceTask],
    s: &StepFile,
) -> (Mesh, Stats) {
    // 1. Collect ALL vertices from ALL faces (CPU)
    let mut all_face_data = Vec::new();
    for task in face_tasks {
        let (verts, surface) = extract_face_geometry(s, task);
        all_face_data.push(FaceGeometry {
            vertices: verts,
            surface,
            task_meta: task.clone(),
        });
    }
    
    // 2. Concatenate into single GPU buffer
    let total_vertices: usize = all_face_data.iter()
        .map(|f| f.vertices.len())
        .sum();
    
    let mut vertex_buffer = Vec::with_capacity(total_vertices);
    let mut face_offsets = Vec::new();
    let mut surface_indices = Vec::new();
    
    for (face_idx, face) in all_face_data.iter().enumerate() {
        face_offsets.push(vertex_buffer.len() as u32);
        vertex_buffer.extend(&face.vertices);
        surface_indices.extend(vec![face_idx as u32; face.vertices.len()]);
    }
    
    // 3. Single GPU upload
    let gpu_verts = device.create_buffer_init(...);
    let gpu_surfaces = device.create_buffer_init(...); // Array of surfaces
    let gpu_surface_indices = device.create_buffer_init(...);
    
    // 4. Single GPU kernel dispatch
    let workgroups = (total_vertices as u32 + 255) / 256;
    cpass.dispatch_workgroups(workgroups, 1, 1);
    
    // 5. Single GPU download
    let results = download_all_results(device, queue).await;
    
    // 6. Unpack results back to individual faces (CPU)
    reconstruct_meshes(results, face_offsets, face_tasks)
}
```

### Phase 2: GPU-Side Triangulation (Expected: Additional 2-3x)

Move CDT triangulation to GPU instead of doing it on CPU:
- Implement Delaunay triangulation in WGSL/compute shader
- Keep edge constraints on GPU
- Only transfer final triangle indices back

**Complexity**: High. CDT is complex. Consider using existing GPU libraries like:
- `delaunator` (port to WGSL)
- Or keep CPU triangulation but batch it

### Phase 3: Async Pipelining (Expected: 1.5-2x)

```rust
// Overlap CPU triangulation with GPU transforms
let (tx_gpu, rx_gpu) = mpsc::channel();
let (tx_cpu, rx_cpu) = mpsc::channel();

// Thread 1: GPU lowering
spawn(move || {
    for batch in face_batches {
        let uvs = gpu_lower_batch(batch);
        tx_gpu.send(uvs).unwrap();
    }
});

// Thread 2: CPU triangulation  
spawn(move || {
    while let Ok(uvs) = rx_gpu.recv() {
        let triangles = cpu_triangulate(uvs);
        tx_cpu.send(triangles).unwrap();
    }
});

// Thread 3: GPU raising
while let Ok(triangles) = rx_cpu.recv() {
    gpu_raise_and_transform_batch(triangles);
}
```

### Phase 4: Persistent GPU Buffers (Expected: 1.2-1.5x)

```rust
struct GpuTriangulator {
    device: Device,
    queue: Queue,
    // Pre-allocated buffers (reuse across faces)
    vertex_buffer: Buffer,  // Max size
    output_buffer: Buffer,
    staging_buffer: Buffer,
    // Pre-compiled pipelines
    lowering_pipeline: ComputePipeline,
    raising_pipeline: ComputePipeline,
}

impl GpuTriangulator {
    fn process_batch(&mut self, faces: &[FaceTask]) -> Mesh {
        // Reuse buffers, no recreation
        self.queue.write_buffer(&self.vertex_buffer, 0, ...);
        // ...
    }
}
```

## Shader Optimizations

### Current Shader Issues

1. **Branching in hot loops**: Your `lower()` function has a huge `switch` statement called per vertex
2. **Unused code paths**: Most faces use 1-2 surface types, but shader compiles all 7 types

### Optimized Shader Strategy

**Specialize shaders per surface type**:
```rust
// Create 6 different pipelines
let plane_pipeline = create_pipeline("lowering_plane.wgsl");
let cylinder_pipeline = create_pipeline("lowering_cylinder.wgsl");
// etc

// Batch faces by surface type
let faces_by_surface: HashMap<SurfaceType, Vec<FaceTask>> = 
    group_by_surface_type(face_tasks);

for (surf_type, faces) in faces_by_surface {
    let pipeline = match surf_type {
        SurfaceType::Plane => &plane_pipeline,
        SurfaceType::Cylinder => &cylinder_pipeline,
        // ...
    };
    process_batch_with_pipeline(faces, pipeline);
}
```

**Specialized plane lowering** (most common):
```wgsl
@compute @workgroup_size(256)
fn lowering_plane(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_vertices)) { return; }
    
    let pos = input_vertices[idx].xyz;
    // Direct computation, no branching
    let uv = (surface.mat_i * vec4<f32>(pos, 1.0)).xy;
    output_vertices[idx] = vec4<f32>(uv, 0.0, 0.0);
}
```

## Implementation Priority

1. **Week 1**: Batching (Phase 1) - This alone should get you to ~10ms
2. **Week 2**: Persistent buffers (Phase 4) - Get to ~7ms  
3. **Week 3**: Specialized shaders - Get to ~5ms
4. **Week 4**: Consider if GPU is even worth it at this point

## Realistic Expectations

GPU acceleration makes sense when:
- ✅ Operation is highly parallel (yours is)
- ✅ Compute is expensive relative to data transfer (yours isn't)
- ❌ Batch sizes are large (yours are tiny)
- ❌ CPU is the bottleneck (your CPU version is already fast)

**Brutal truth**: For your workload, GPU might never beat well-optimized CPU code. Consider:
- Your CPU version with `triangulate6` (with caching) might be the winner
- Focus optimization efforts there instead
- GPU makes sense for rendering, not necessarily geometry processing with tiny batches

## PollType Fix

```rust
// Old (wgpu <27)
device.poll(wgpu::PollType::Wait { 
    submission_index: None, 
    timeout: None 
});

// New (wgpu 27+)
device.poll(wgpu::Maintain::Wait);
// or
device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
```