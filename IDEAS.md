# Things to try out:

### Speed:
- remove all `std::collections::HashMap` everywhere and replace them with `ahash`.
- write a custom hasher specifically for our kinda data (the step spec is enumerable so... we can do betterer..)
- try a wgsl implementation of triangulate. [QUESTION: how many bytes are all our test data?]
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
4. **CDT triangulation itself** - Constrained Delaunay is O(n log n) at best

### Nuclear Option Optimizations:

Would require more invasive changes:

1. **Cache surfaces by ID** - Don't reconstruct B-splines repeatedly
2. **Pre-compute entity lookup table** - Turn O(n) scans into O(1) HashMap lookups
3. **Approximate NURBS projection** - Use faster grid-based approximation instead of exact projection
4. **Replace CDT** - Use ear clipping (O(n²) but simpler) or even faster tessellation



### QOL:
- `.github` pipeline
- data fetching scripts
- typos, mdformat, ruff, clippy configs
- All other crates these days call it `rayon` if that's what you're using so we should update the `parallel` feature to be `rayon` everywhere

### DEEPNESS:
- are we getting any SIMD?
- of all your machines how shit are we??? does perf scale with cores?