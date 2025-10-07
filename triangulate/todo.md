# Triangulate - WGPU Implementation Roadmap

## Immediate Correctness Issues (Priority 1)

### Root Cause: Architectural Flaws in GPU Usage
The current GPU implementation suffers from a fundamental architectural problem:
- It re-does work the CPU has already done correctly
- It uses GPU for lowering/raising operations that should be CPU-only
- This causes geometric projection failures resulting in "wheel" artifacts

### Phase 1: Fix Core Architecture
- [ ] **Analyze current wgpu_impl.rs** to understand where the lowering/raising operations are happening
- [ ] **Identify the "prepare()" step** that's missing from the GPU pipeline (crucial for complex surfaces like spheres/tori)
- [ ] **Verify the GpuSurface conversion** is working correctly with proper transformation matrices
- [ ] **Document all places where the GPU is incorrectly re-processing CPU-work**

### Phase 2: Implement Corrected Approach
Based on the corrective action plan in correctness-ideas.md:

#### Step 1: Create Simplified Transformation Shader
- [ ] Create new file: `./triangulate/src/wgpu_triangulate/transform_mesh.wgsl`
- [ ] Implement focused shader that only applies transformations (no lowering/raising)
- [ ] Define `GpuTemplateVertex`, `GpuTransform`, and `GpuOutputVertex` structs
- [ ] Implement compute shader logic for parallel vertex transformation

#### Step 2: Refactor wgpu_impl.rs
- [ ] Remove faulty lowering/raising functions from current implementation
- [ ] Create new `gpu_transform_mesh` function that uses the transformation shader
- [ ] Implement proper data structures: `GpuTransform`, `GpuVertex`
- [ ] Set up correct GPU buffers and pipeline for transformation operations
- [ ] Handle GPU-CPU data transfer correctly

#### Step 3: Correct Main Triangulation Flow
- [ ] Modify `triangulate_faces` to use CPU for triangulation and GPU only for transformations
- [ ] Ensure CPU generates correct "template" meshes first
- [ ] Pass template meshes to GPU for parallel transformation application
- [ ] Handle triangle index replication on CPU (extremely fast operation)

## Validation and Testing (Priority 2)

### Correctness Validation
- [ ] Create comparison tests between CPU and corrected GPU outputs
- [ ] Verify that "wheel" artifacts are eliminated
- [ ] Test with complex geometries (spheres, tori, cylinders)
- [ ] Validate that all surface types produce identical results

### Unit Testing
- [ ] Add unit tests for the new transformation shader
- [ ] Test edge cases in transformation logic
- [ ] Verify normal transformations are correct (inverse transpose)
- [ ] Test with various transformation matrix types

## Performance Optimization (Priority 3)

Once correctness is established:

### Batch Processing
- [ ] Implement batched processing of all faces in single GPU operation
- [ ] Eliminate per-face CPU-GPU transfers
- [ ] Process all transformations in one GPU dispatch

### Memory Management
- [ ] Reuse GPU buffers across multiple calls
- [ ] Implement persistent GPU resources to avoid reallocation
- [ ] Optimize data marshalling between CPU and GPU

### Shader Optimization
- [ ] Profile and optimize the transformation shader
- [ ] Minimize redundant computations
- [ ] Optimize memory access patterns

## Benchmarking and Verification

### Performance Goals
- [ ] Achieve 3-10x speedup over CPU implementation (not 178x slower)
- [ ] Process all faces in single GPU operation rather than per-face transfers
- [ ] Minimize GPU starvation by batching work effectively

### Validation Metrics
- [ ] Output must match CPU implementation exactly
- [ ] Handle all STEP file types correctly
- [ ] Maintain proper error handling and reporting
- [ ] No geometric artifacts or distortions

## Code Quality Improvements

### Error Handling
- [ ] Replace `.expect()` with proper error propagation
- [ ] Implement consistent error handling strategy
- [ ] Improve panic handling in parallel code

### Code Organization
- [ ] Make wgpu implementation completely sandboxed in wgpu_triangulate/**/* files
- [ ] Remove code duplication in triangulation functions
- [ ] Use type aliases for complex types
- [ ] Replace magic numbers with named constants

## Long-term Architecture Considerations

### GPU-Only Pipeline (Future)
Consider if a fully GPU-based pipeline would be beneficial:
- Pros: Potentially even better performance for large datasets
- Cons: Much more complex implementation, harder debugging
- Decision: Focus on hybrid CPU-GPU approach for now (CPU for complex logic, GPU for parallel math)

### Feature Completeness
- [ ] Ensure all surface types work correctly on GPU
- [ ] Handle boundary conditions properly
- [ ] Support all STEP file features that CPU implementation supports