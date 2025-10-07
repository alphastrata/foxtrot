# Triangulate - Batch Processing WGPU Performance Optimization

## Objective
Create batch triangulation benchmark to showcase where GPU acceleration provides the most benefit by processing multiple STEP files simultaneously.

## Implementation Plan

### Phase 1: Data Collection and Setup
- [ ] Scan `./examples` directory for all `.step` and `.stp` files
- [ ] Create a batch processing function `wgpu_triangulate_batch(&[Vec<u8>]) -> Vec<(Mesh, Stats)>`
- [ ] Implement parallel loading of STEP file data

### Phase 2: Batch Processing Implementation 
- [ ] Create new benchmark file `benches/batch_triangulation.rs`
- [ ] Implement CPU-only batch version using triangulate4 as baseline
- [ ] Implement GPU batch version using unified GPU resources
- [ ] Design efficient data marshalling for batch operations

### Phase 3: GPU Resource Optimization
- [ ] Set up persistent GPU resources that can be reused across files
- [ ] Optimize buffer allocation and management for batch processing
- [ ] Minimize GPU initialization overhead when processing multiple files
- [ ] Implement efficient command buffer batching

### Phase 4: Performance Testing
- [ ] Benchmark CPU vs GPU batch performance on various file collections
- [ ] Measure scaling behavior with increasing number of files
- [ ] Profile memory usage and GPU utilization
- [ ] Test with different file types (complex vs simple geometries)

### Phase 5: Analysis and Optimization
- [ ] Identify bottlenecks in batch processing pipeline
- [ ] Optimize the batch processing for maximum GPU utilization
- [ ] Determine optimal batch sizes for different scenarios
- [ ] Document performance gains and use cases where GPU excels

## Expected Outcomes
- GPU should significantly outperform CPU when processing multiple files due to parallelization
- GPU overhead becomes negligible when processing multiple files simultaneously
- Batch processing should demonstrate clear performance advantages of GPU implementation