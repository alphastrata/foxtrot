# Things to try out:


### misc:
- We need more tests in the triangulation, and cdt area.
- the wgpu code is currently wrong (at triangulating) so the tests for the above should be a refrerence for future development against (despite wgpu being much slower than the CPU for triangulation, I still believe in sufficently large batch jobs, and to alleviate cpu pressure! it can, or at least could be useful.)
- All other crates these days call it `rayon` if that's what you're using so we should update the `parallel` feature to be `rayon` everywhere
- rename unwisely named fucking_pythagoras.rs to something mose sensible... move ALL but the original fn triangulate() so that's all the numbered ones into that new file. (the wgpu ones can stay where they are.)


### Speed:
- remove all `std::collections::HashMap` everywhere and replace them with `ahash`.
- write a custom hasher specifically for our kinda data (the step spec is enumerable so... we can do betterer..)
- `libm` in place of all existing math stuff, everywhere.
- `samply` the `thumbnailer` or similar so we can find the exact hottest loops (undoubtedly within meshing)

Let's dig deeper. The real bottleneck is likely inside the triangulation itself. Here's a more aggressive optimization approach:# Additional Aggressive Optimizations

The artifact above has these key improvements:

## 1. **Removed Redundant Allocations**
- Pre-sized buffers (`with_capacity`) based on typical face geometry
- Avoided cloning `pts` for panic recovery
- Used atomic counters instead of collecting stats objects

## 2. **Removed Panic Catching**
- `catch_unwind` is expensive (saves/restores panic hooks)
- Replaced with normal error propagation
- ~10-20% speedup on hot path

## 3. **Streamlined Face Collection**
- Single-pass extraction without intermediate vectors
- Direct iteration instead of collecting then mapping

## 4. **Reduced Memory Pressure**
- Faces processed independently = better cache locality
- No template mesh cloning
- Mesh combine is efficient (just concatenates)

## But if you want **TRULY FAST**, here's what's *really* killing performance:

### The Real Bottlenecks:

1. **`face_bound()` and `edge_loop()`** - These walk linked lists of edges, doing O(n) lookups for each entity
2. **`get_surface()`** - Constructs B-spline surfaces from scratch every time
3. **`surf.lower_verts()`** - Projects 3D→2D, likely doing iterative Newton-Raphson for NURBS

### Nuclear Option Optimizations:

Would require more invasive changes:

1. **Cache surfaces by ID** - Don't reconstruct B-splines repeatedly
2. **Pre-compute entity lookup table** - Turn O(n) scans into O(1) HashMap lookups
3. **Approximate NURBS projection** - Use faster grid-based approximation instead of exact projection
4. **Alternatives to CDT** - Offer (--feature="ear-clipping") Use ear clipping (O(n²) but simpler) or even faster tessellation

Mixed ideas;
# 1. Functions to Improve

## High-Priority Optimizations

### `edge_loop` and `edge_loop2`
**Current Issue**: Allocates new vectors repeatedly, performs redundant lookups
**Method**: 
- Pre-allocate output vector with estimated capacity
- Use a single pass through edges without repeated `pop()` operations
- Cache `EdgeCurve` lookups in a local hashmap
**Justification**: Called for every face boundary, hot path in triangulation

### `face_bound` and `face_bound2`
**Current Issue**: Allocates and potentially reverses entire contour vectors
**Method**:
- Build contours in correct direction initially based on `orientation` flag
- Pass a "reverse" flag down to `edge_loop` to build backwards when needed
**Justification**: Eliminates O(n) reversal operation per boundary

### `curve` and `curve2`
**Current Issue**: Repetitive pattern matching, doesn't leverage cache effectively
**Method**:
- Pre-compute and cache all curve objects during initial pass
- Flatten nested `SurfaceCurve`/`SeamCurve` indirection during cache build
**Justification**: Called once per edge, but pattern matching overhead adds up

### `control_points_1d` and `control_points_2d`
**Current Issue**: Allocates many small vectors, iterates multiple times
**Method**:
- Use `flat_map` to build single flat iterator
- Collect directly into pre-sized vector
**Justification**: Called for every NURBS/BSpline surface

### `advanced_face_to_mesh` / `triangulate_single_face`
**Current Issue**: Creates temporary `pts` clone for panic handling
**Method**:
- Remove panic catching entirely - CDT library should be robust enough
- If panics occur, fix them in the CDT library rather than catching
**Justification**: Panic overhead is significant, and catching panics is a code smell

### `Mesh::combine`
**Current Issue**: Sequential extension of vectors
**Method**:
```rust
pub fn combine_many(meshes: Vec<Self>) -> Self {
    let total_verts: usize = meshes.iter().map(|m| m.verts.len()).sum();
    let total_tris: usize = meshes.iter().map(|m| m.triangles.len()).sum();
    
    let mut result = Mesh {
        verts: Vec::with_capacity(total_verts),
        triangles: Vec::with_capacity(total_tris),
    };
    
    for mesh in meshes {
        let offset = result.verts.len() as u32;
        result.verts.extend(mesh.verts);
        result.triangles.extend(
            mesh.triangles.into_iter()
                .map(|mut t| { t.verts.add_scalar_mut(offset); t })
        );
    }
    result
}
```
**Justification**: Reduces allocations from O(n) to O(1) when combining many meshes


### QOL:
- `.github` pipeline
- data fetching scripts
- typos, mdformat, ruff, clippy configs

### DEEPNESS:
- are we getting any SIMD?
- of all your machines how shit are we??? does perf scale with cores?


~- try a wgsl implementation of triangulate. [QUESTION: how many bytes are all our test data?]~DONE
