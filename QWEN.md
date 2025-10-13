# Foxtrot Project Context

Foxtrot is a Rust-based toolkit for working with STEP (Standard for the Exchange of Product Data) files, which are used in CAD (Computer-Aided Design) applications. The project provides tools for parsing, triangulating, and rendering STEP files.

## Project Structure

The project is organized as a Cargo workspace with multiple crates:

- `cdt`: Constrained Delaunay triangulation library (standalone)
- `express`: Parser for EXPRESS schema files and code generation system
- `step`: Auto-generated STEP file parser (takes a long time to compile)
- `triangulate`: Converts STEP files into triangle meshes using `cdt` as core
- `nurbs`: NURBS/B-spline algorithms used by `triangulate`
- `gui`: GUI for rendering STEP files using WebGPU
- `wasm`: Scaffolding for running in browser with WebAssembly
- `examples/thumbnailer`: Command-line utility for generating PNG thumbnails from STEP files

## Key Technologies

- **Rust**: Main programming language
- **WebGPU**: Graphics API for rendering (through `wgpu` crate)
- **Rayon**: Parallel processing library
- **nalgebra-glm**: Linear algebra library for 3D math
- **AHASH**: Fast hashing library for collections

## Building and Running

### Prerequisites
- Install Rust and Cargo: https://doc.rust-lang.org/cargo/getting-started/installation.html

### Quick Start
```bash
# Run the GUI with a sample STEP file
cargo run --release -- examples/cube_hole.step
```

### WebAssembly Demo
```bash
# Requires wasm-pack installation
cd wasm
wasm-pack build --target no-modules
python3 -m http.server --directory deploy
```

### Thumbnailer Usage
```bash
# Using the default Foxtrot engine
cargo run --bin step_thumbnailer -- -i examples/cube_hole.step -o cube_hole.png --size 512

# Using the OCCT engine (requires `occt` feature)
cargo run --bin step_thumbnailer --features occt -- -i examples/cube_hole.step -o cube_hole.png --size 512 --engine occt
```

## Features and Compilation

The triangulate crate supports several optional features:
- `rayon`: Enables parallel processing capabilities
- `wgpu`: Enables GPU acceleration for triangulation
- `test`: Enables test functionality (includes `wgpu` and `rayon`)
- `benchmarks`: Enables benchmark functionality (includes `wgpu` and `rayon`)

## Development Notes

### Code Generation
The `step/src/ap214.rs` file is automatically generated from `10303-214e3-aim-long.exp`. To regenerate:
```bash
cargo run --release --example gen_exp -- path/to/APs/10303-214e3-aim-long.exp step/src/ap214.rs
```

### Benchmarking
Benchmarks are implemented using the Criterion framework and can be run with:
```bash
cargo bench -p triangulate --features rayon,wgpu,benchmarks
```

### Testing
Unit tests can be run with:
```bash
cargo test -p triangulate --features rayon,wgpu
```

## Project Status and Limitations

Foxtrot is described as a "proof-of-concept demo, not an industrial-strength CAD kernel." It may not work for all STEP models and has known limitations in triangulation capabilities.

## Key Modules in triangulate Crate

- `triangulate::`: Main triangulation implementation
- `triangulate::historical_triangulations`: Previous triangulation implementations kept for benchmarking
- `wgpu_triangulate::`: GPU-accelerated triangulation implementations
- `wgpu_triangulate::triangulation_utils`: Utility functions for GPU triangulation
- `wgpu_triangulate::cached_triangulation`: Cached triangulation implementations
- `wgpu_triangulate::batch_triangulation`: Batch processing triangulation implementations

## Development Conventions

- Code follows Rust 2024 edition conventions
- Extensive use of conditional compilation (`#[cfg(...)]`) for features
- Benchmarking uses the Criterion framework
- Comprehensive error handling with `thiserror`
- Logging through the `log` crate
- Parallel processing through `rayon` when enabled