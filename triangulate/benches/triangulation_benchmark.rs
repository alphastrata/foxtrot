use criterion::{Criterion, criterion_group, criterion_main};
use std::fs;
use step::step_file::StepFile;

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
            // CPU-only benchmark group for historical implementations
            {
                let mut group = c.benchmark_group("cpu_triangulators");

                // Original triangulation from historical module
                group.bench_function("triangulate-original", |b| {
                    b.iter(|| {
                        _ = triangulate::triangulate::historical_triangulations::triangulate(
                            &step_file,
                        );
                    });
                });

                {
                    group.bench_function("triangulate-rayon-v2", |b| {
                        b.iter(|| {
                            _ = triangulate::triangulate::historical_triangulations::triangulate2(
                                &step_file,
                            );
                        });
                    });

                    group.bench_function("triangulate-rayon-v3", |b| {
                        b.iter(|| {
                            _ = triangulate::triangulate::historical_triangulations::triangulate3(
                                &step_file,
                            );
                        });
                    });
                }

                // Cached versions (if rayon is enabled)
                {
                    group.bench_function("triangulate-rayon-v6", |b| {
                        b.iter(|| {
                            _ = triangulate::triangulate::historical_triangulations::triangulate6(
                                &step_file,
                            );
                        });
                    });
                }

                // WGPU-based versions from historical
                {
                    group.bench_function("triangulate-wgpu-v5", |b| {
                        b.iter(|| {
                            _ = triangulate::triangulate::historical_triangulations::triangulate5(
                                &step_file,
                            );
                        });
                    });
                }

                group.finish();
            }

            // WGPU benchmark group
            {
                let mut group = c.benchmark_group("wgpu_triangulators");
                group.bench_function("triangulate-wgpu", |b| {
                    b.iter(|| {
                        _ = triangulate::wgpu_triangulate::wgpu_triangulate(&step_file);
                    });
                });
                group.bench_function("triangulate-wgpu2", |b| {
                    b.iter(|| {
                        _ = triangulate::wgpu_triangulate::triangulate(&step_file);
                    });
                });

                group.finish();
            }
        } else {
            eprintln!("Warning: STEP file is empty, benchmarking will not be meaningful");
        }
    }
}

fn benchmark_batch_triangulate(c: &mut Criterion) {
    // WGPU batch benchmark group
    {
        let mut group = c.benchmark_group("wgpu_batch_triangulators");
        group.bench_function("triangulate-batch", |b| {
            b.iter(|| {
                _ = triangulate::wgpu_triangulate::wgpu_triangulate_batch_from_examples();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark_triangulate, benchmark_batch_triangulate);
criterion_main!(benches);
