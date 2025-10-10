use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::fs;
use std::path::Path;
use step::step_file::StepFile;

use triangulate::wgpu_triangulate::{wgpu_triangulate, wgpu_triangulate_batch_from_examples};

/// Find all STEP files in the examples directory
fn find_step_files() -> Vec<String> {
    let mut step_files_paths = Vec::new();

    let examples_paths = ["../examples", "./examples"];
    for examples_path in &examples_paths {
        let path = Path::new(examples_path);
        if path.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_file() {
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase());
                    if ext == Some("step".to_string()) || ext == Some("stp".to_string()) {
                        step_files_paths.push(path.to_string_lossy().to_string());
                    }
                }
            }
            break;
        }
    }

    step_files_paths
}

/// Benchmark individual GPU triangulation (one device per file)
fn benchmark_individual_gpu_triangulation(c: &mut Criterion) {
    let mut step_files = find_step_files();

    if step_files.is_empty() {
        eprintln!("No STEP files found in examples directories for benchmarking");
        return;
    }

    // Increase number of files by duplicating existing ones if needed
    // This allows us to test with more data without needing actual files
    let target_batch_size = 10.min(step_files.len()); // Cap at 10 for faster iteration
    while step_files.len() < target_batch_size {
        step_files.extend(step_files.clone());
        if step_files.len() > target_batch_size {
            step_files.truncate(target_batch_size);
            break;
        }
    }

    // Configure Criterion with sample size
    let mut group = c.benchmark_group("Individual GPU Triangulation");
    group.sample_size(10.min(step_files.len())); // Cap at 10 runs
    group.throughput(Throughput::Elements(step_files.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("Individual GPU", step_files.len()),
        &step_files,
        |b, files| {
            b.iter(|| {
                files
                    .iter()
                    .map(|path| {
                        if let Ok(contents) = fs::read(path) {
                            let flattened = StepFile::strip_flatten(&contents);
                            let step_file = StepFile::parse(&flattened);
                            wgpu_triangulate(&step_file)
                        } else {
                            (
                                triangulate::mesh::Mesh::default(),
                                triangulate::stats::Stats::default(),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            });
        },
    );

    group.finish();
}

/// Benchmark batch GPU triangulation (single device for all files)
fn benchmark_batch_gpu_triangulation(c: &mut Criterion) {
    let mut step_files = find_step_files();

    if step_files.is_empty() {
        eprintln!("No STEP files found in examples directories for benchmarking");
        return;
    }
    // Increase number of files by duplicating existing ones if needed
    // This allows us to test with more data without needing actual files
    let target_batch_size = 10.min(step_files.len()); // Cap at 10 for faster iteration
    while step_files.len() < target_batch_size {
        step_files.extend(step_files.clone());
        if step_files.len() > target_batch_size {
            step_files.truncate(target_batch_size);
            break;
        }
    }

    // Configure Criterion with sample size
    let mut group = c.benchmark_group("Batch GPU Triangulation");
    group.sample_size(10.min(step_files.len())); // Cap at 10 runs
    group.throughput(Throughput::Elements(step_files.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("Batch GPU", step_files.len()),
        &step_files,
        |b, _files| {
            b.iter(|| triangulate::wgpu_triangulate::wgpu_triangulate_batch_from_examples());
        },
    );

    group.finish();
}

/// Compare individual vs batch performance
fn benchmark_individual_vs_batch(c: &mut Criterion) {
    let mut step_files = find_step_files();

    if step_files.is_empty() {
        eprintln!("No STEP files found in examples directories for benchmarking");
        return;
    }

    // Increase number of files by duplicating existing ones if needed
    // This allows us to test with more data without needing actual files
    let target_batch_size = 10.min(step_files.len()); // Cap at 10 for faster iteration
    while step_files.len() < target_batch_size {
        step_files.extend(step_files.clone());
        if step_files.len() > target_batch_size {
            step_files.truncate(target_batch_size);
            break;
        }
    }

    let mut group = c.benchmark_group("Batch Comparison");
    group.sample_size(10.min(step_files.len())); // Cap at 10 runs
    group.throughput(Throughput::Elements(step_files.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("Individual GPU", step_files.len()),
        &step_files,
        |b, files| {
            b.iter(|| {
                files
                    .iter()
                    .map(|path| {
                        if let Ok(contents) = fs::read(path) {
                            let flattened = StepFile::strip_flatten(&contents);
                            let step_file = StepFile::parse(&flattened);
                            wgpu_triangulate(&step_file)
                        } else {
                            (
                                triangulate::mesh::Mesh::default(),
                                triangulate::stats::Stats::default(),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Batch GPU", step_files.len()),
        &step_files,
        |b, _files| {
            b.iter(|| wgpu_triangulate_batch_from_examples());
        },
    );

    group.finish();
}

criterion_group!(individual_benches, benchmark_individual_gpu_triangulation);
criterion_group!(batch_benches, benchmark_batch_gpu_triangulation);
criterion_group!(comparison_benches, benchmark_individual_vs_batch);

criterion_main!(individual_benches, batch_benches, comparison_benches);
