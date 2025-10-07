use ::triangulate::triangulate::wgpu_impl;
use criterion::{Criterion, criterion_group, criterion_main};
use std::fs;
use step::step_file::StepFile;
use triangulate::triangulate;
use log::warn;

fn benchmark_triangulate(c: &mut Criterion) {
    // Load a test STEP file for benchmarking by reading it in the benchmark setup
    let examples_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("examples");

    // Find and read a complex STEP file for benchmarking
    let step_file_content: Option<Vec<u8>> =
        fs::read(examples_path.join("21492_8bd34fc1_15.stp")).ok();
    if step_file_content.is_none() {
        panic!(
            "Benchmark STEP file not found in examples directory, consult the TESTING.md docs to get this step file!"
        );
    }

    if let Some(content) = step_file_content {
        // Preprocess the content as required by StepFile::parse
        let flattened = StepFile::strip_flatten(&content);
        let step_file = StepFile::parse(&flattened);

        // Make sure the STEP file has content for meaningful benchmarking
        if !step_file.0.is_empty() {
            // Comparison benchmark group
            #[cfg(feature = "rayon")]
            {
                let mut group = c.benchmark_group("triangulators");
                group.bench_function("triangulate-original", |b| {
                    b.iter(|| {
                        _ = triangulate::triangulate(&step_file);
                    });
                });
                group.bench_function("triangulate-2", |b| {
                    b.iter(|| {
                        _ = triangulate::triangulate2(&step_file);
                    });
                });
                group.bench_function("triangulate-3", |b| {
                    b.iter(|| {
                        _ = triangulate::triangulate3(&step_file);
                    });
                });
                group.bench_function("triangulate-4", |b| {
                    b.iter(|| {
                        _ = triangulate::triangulate4(&step_file);
                    });
                });
                group.bench_function("triangulate-5", |b| {
                    b.iter(|| {
                        _ = triangulate::cached_triangulation::triangulate5(&step_file);
                    });
                });
                group.bench_function("triangulate-6", |b| {
                    b.iter(|| {
                        _ = triangulate::cached_triangulation::triangulate6(&step_file);
                    });
                });
                group.bench_function("triangulate-wgpu", |b| {
                    b.iter(|| {
                        _ = wgpu_impl::triangulate(&step_file);
                    });
                });

                group.finish();
            }
        } else {
            warn!("Warning: STEP file is empty, benchmarking will not be meaningful");
        }
    }
}

criterion_group!(benches, benchmark_triangulate);
criterion_main!(benches);
