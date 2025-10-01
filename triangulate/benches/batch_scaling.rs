use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::fs;
use std::path::Path;
use ahash::AHashMap;
use step::{
    ap214::*,
    id::Id,
    step_file::{FromEntity, StepFile},
};
use nalgebra_glm as glm;
use triangulate::{triangulate::{build_transform_stack, collect_faces_from_brep, presentation_style_color, transform_stack_roots}, wgpu_triangulate::wgpu_utils};

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

/// Benchmark batch processing with various sizes to show scaling behavior
fn benchmark_batch_scaling(c: &mut Criterion) {
    let original_step_files = find_step_files();
    
    if original_step_files.is_empty() {
        eprintln!("No STEP files found in examples directories for benchmarking");
        return;
    }
    
    let batch_sizes = [10,25,50,100];
    let mut group = c.benchmark_group("Batch Scaling Performance");
    
    for &batch_size in &batch_sizes {
        // Create a batch of the required size by duplicating files
        let mut step_files = Vec::new();
        for i in 0..batch_size {
            step_files.push(original_step_files[i % original_step_files.len()].clone());
        }
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.sample_size(10); // Criterion requires at least 10 samples
        group.bench_with_input(
            BenchmarkId::new("Batch GPU", batch_size),
            &step_files,
            |b, _files| {
                // Create a temporary function that replicates the batch processing logic
                // but for a specific set of files
                b.iter(|| {
                    // Create a single GPU device for the entire batch
                    let (device, queue) = match wgpu_utils::create_wgpu_device() {
                        Ok(dq) => dq,
                        Err(_) => return,
                    };

                    // Process each file using the shared GPU device
                    for path in &step_files {
                        if let Ok(contents) = fs::read(path) {
                            let flattened = StepFile::strip_flatten(&contents);
                            let step_file = StepFile::parse(&flattened);
                            
                            // Build face tasks for this specific file using the same logic as wgpu_triangulate
                            let brep_colors: AHashMap<_, glm::DVec3> =
                                step_file.0.iter()
                                    .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
                                    .flat_map(|m| m.items.iter())
                                    .filter_map(|item| step_file.entity(item.cast::<StyledItem_>()))
                                    .filter_map(|styled| {
                                        if styled.styles.len() == 1 {
                                            presentation_style_color(&step_file, styled.styles[0]).map(|c| (styled.item, c))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();

                            let mut transform_stack = build_transform_stack(&step_file, false);
                            let mut roots = transform_stack_roots(&transform_stack);
                            if roots.len() > 1 {
                                transform_stack = build_transform_stack(&step_file, true);
                                roots = transform_stack_roots(&transform_stack);
                            }

                            let mut todo: Vec<_> = roots.into_iter().map(|v| (v, glm::DMat4::identity())).collect();
                            let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
                            for (r1, r2) in
                                step_file.0.iter()
                                    .filter_map(ShapeRepresentationRelationship_::try_from_entity)
                                    .map(|e| (e.rep_1, e.rep_2))
                            {
                                shape_rep_relationship.entry(r1).or_default().push(r2);
                            }

                            let mut to_mesh: AHashMap<Id<_>, Vec<glm::DMat4>> = AHashMap::new();
                            while let Some((id, mat)) = todo.pop() {
                                for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
                                    todo.push((*child, mat));
                                }
                                if let Some(children) = transform_stack.get(&id) {
                                    for (child, next_mat) in children {
                                        todo.push((*child, mat * next_mat));
                                    }
                                } else {
                                    let items = match &step_file[id] {
                                        Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                                        Entity::ShapeRepresentation(b) => &b.items,
                                        Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                                        _ => continue,
                                    };

                                    for m in items.iter() {
                                        if matches!(
                                            &step_file[*m],
                                            Entity::ManifoldSolidBrep(_)
                                                | Entity::BrepWithVoids(_)
                                                | Entity::ShellBasedSurfaceModel(_)
                                        ) {
                                            to_mesh.entry(*m).or_default().push(mat);
                                        }
                                    }
                                }
                            }

                            if to_mesh.is_empty() {
                                to_mesh =
                                    step_file.0.iter()
                                        .enumerate()
                                        .filter(|(_i, e)| {
                                            matches!(
                                                e,
                                                Entity::ManifoldSolidBrep(_)
                                                    | Entity::BrepWithVoids(_)
                                                    | Entity::ShellBasedSurfaceModel(_)
                                            )
                                        })
                                        .map(|(i, _e)| (Id::new(i), vec![glm::DMat4::identity()]))
                                        .collect();
                            }

                            // Extract all face IDs with metadata (no deep copies)
                            let face_tasks: Vec<triangulate::triangulate::FaceTask> = to_mesh
                                .into_iter()
                                .flat_map(|(brep_id, mats)| {
                                    let color = brep_colors
                                        .get(&brep_id)
                                        .copied()
                                        .unwrap_or(glm::DVec3::new(0.5, 0.5, 0.5));

                                    collect_faces_from_brep(&step_file, brep_id)
                                        .into_iter()
                                        .map(move |(face_id, flip)| triangulate::triangulate::FaceTask {
                                            face_id,
                                            transforms: mats.clone(),
                                            color,
                                            flip_normal: flip,
                                        })
                                })
                                .collect();
                            
                            // Use the triangulate function with the shared GPU device
                            let (_mesh, _stats) = wgpu_utils::triangulate_faces_with_device(&step_file, &face_tasks, &device, &queue);
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(scaling_benches, benchmark_batch_scaling);
criterion_main!(scaling_benches);
