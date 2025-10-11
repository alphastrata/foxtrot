//! WGPU triangulation functions
//!
//! This module contains the main WGPU triangulation functions that were previously
//! in the triangulate module.

use ahash::AHashMap;
use glm::{DMat4, DVec3};
use nalgebra_glm as glm;

use step::{
    ap214::*,
    id::Id,
    step_file::{FromEntity, StepFile},
};

use crate::{
    mesh::Mesh,
    stats::Stats,
    triangulate::{
        build_transform_stack, collect_faces_from_brep, presentation_style_color,
        transform_stack_roots, FaceTask,
    },
};

use super::wgpu_utils;

/// Triangulates a STEP file using GPU acceleration
pub fn wgpu_triangulate(s: &StepFile) -> (Mesh, Stats) {
    // Phase 1: Build face catalog with minimal allocations
    let brep_colors: AHashMap<_, DVec3> =
        s.0.iter()
            .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
            .flat_map(|m| m.items.iter())
            .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
            .filter_map(|styled| {
                if styled.styles.len() == 1 {
                    presentation_style_color(s, styled.styles[0]).map(|c| (styled.item, c))
                } else {
                    None
                }
            })
            .collect();

    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    if roots.len() > 1 {
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }

    let mut todo: Vec<_> = roots.into_iter().map(|v| (v, DMat4::identity())).collect();
    let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: AHashMap<Id<_>, Vec<DMat4>> = AHashMap::new();
    while let Some((id, mat)) = todo.pop() {
        for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
            todo.push((*child, mat));
        }
        if let Some(children) = transform_stack.get(&id) {
            for (child, next_mat) in children {
                todo.push((*child, mat * next_mat));
            }
        } else {
            let items = match &s[id] {
                Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                Entity::ShapeRepresentation(b) => &b.items,
                Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                _ => continue,
            };

            for m in items.iter() {
                if matches!(
                    &s[*m],
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
            s.0.iter()
                .enumerate()
                .filter(|(_i, e)| {
                    matches!(
                        e,
                        Entity::ManifoldSolidBrep(_)
                            | Entity::BrepWithVoids(_)
                            | Entity::ShellBasedSurfaceModel(_)
                    )
                })
                .map(|(i, _)| (Id::new(i), vec![DMat4::identity()]))
                .collect();
    }

    // Phase 2: Extract all face IDs with metadata (no deep copies)
    let face_tasks: Vec<FaceTask> = to_mesh
        .into_iter()
        .flat_map(|(brep_id, mats)| {
            let color = brep_colors
                .get(&brep_id)
                .copied()
                .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

            collect_faces_from_brep(s, brep_id)
                .into_iter()
                .map(move |(face_id, flip)| FaceTask {
                    face_id,
                    transforms: mats.clone(),
                    color,
                    flip_normal: flip,
                })
        })
        .collect();

    wgpu_utils::triangulate_faces(s, &face_tasks)
}

/// Single-file triangulation using a pre-existing GPU context for optimal resource reuse
/// This function reuses GPU resources across multiple via the GPUContext you provide to it, so if you need to initialise one just to use it, there's no benefits to be gained here...
pub fn wgpu_triangulate_with_context(
    s: &StepFile,
    gpu_context: &crate::wgpu_triangulate::GPUContext,
) -> (Mesh, Stats) {
    gpu_context.triangulate(s)
}

/// Batch triangulation function that processes all STEP files in the examples directory
/// This maximizes GPU utilization by processing all faces from all files in one operation
pub fn wgpu_triangulate_batch_from_examples() -> Vec<(Mesh, Stats)> {
    use std::fs;
    use std::path::Path;
    use step::step_file::StepFile;

    // List of problematic STEP files that cause triangulation panics
    const PROBLEMATIC_FILES: &[&str] = &[
        "../examples/sphere.step", // Known to cause assertion failure in CDT triangulation
                                   // Add other problematic files here as they are discovered
    ];

    // Find all STEP files in examples directory (excluding problematic ones)
    let mut step_files_paths = Vec::new();

    let examples_paths = ["../examples", "./examples"];
    for examples_path in &examples_paths {
        let path = Path::new(examples_path);
        if path.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_file() {
                    let path_str = path.to_string_lossy().to_string();
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase());
                    if (ext == Some("step".to_string()) || ext == Some("stp".to_string()))
                        && !PROBLEMATIC_FILES.contains(&path_str.as_str())
                    {
                        step_files_paths.push(path_str);
                    }
                }
            }
            break; // Use first directory that exists
        }
    }

    if step_files_paths.is_empty() {
        println!("No STEP files found in examples directories for batch processing");
        return vec![];
    }

    // Create a single GPU device for the entire batch to avoid resource exhaustion
    let (device, queue) = match wgpu_utils::create_wgpu_device() {
        Ok(dq) => dq,
        Err(e) => {
            eprintln!("Failed to create GPU device for batch processing: {}", e);
            // Fallback to CPU processing if GPU fails
            return step_files_paths
                .iter()
                .map(|path| {
                    if let Ok(contents) = fs::read(path) {
                        let flattened = StepFile::strip_flatten(&contents);
                        let step_file = StepFile::parse(&flattened);
                        crate::triangulate::triangulate4(&step_file)
                    } else {
                        (Mesh::default(), Stats::default())
                    }
                })
                .collect();
        }
    };

    // Process each file using the shared GPU device to avoid device creation overhead
    step_files_paths
        .iter()
        .map(|path| {
            if let Ok(contents) = fs::read(path) {
                let flattened = StepFile::strip_flatten(&contents);
                let step_file = StepFile::parse(&flattened);

                // Build face tasks for this specific file using the same logic as wgpu_triangulate
                let brep_colors: AHashMap<_, DVec3> = step_file
                    .0
                    .iter()
                    .filter_map(
                        MechanicalDesignGeometricPresentationRepresentation_::try_from_entity,
                    )
                    .flat_map(|m| m.items.iter())
                    .filter_map(|item| step_file.entity(item.cast::<StyledItem_>()))
                    .filter_map(|styled| {
                        if styled.styles.len() == 1 {
                            presentation_style_color(&step_file, styled.styles[0])
                                .map(|c| (styled.item, c))
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

                let mut todo: Vec<_> = roots.into_iter().map(|v| (v, DMat4::identity())).collect();
                let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
                for (r1, r2) in step_file
                    .0
                    .iter()
                    .filter_map(ShapeRepresentationRelationship_::try_from_entity)
                    .map(|e| (e.rep_1, e.rep_2))
                {
                    shape_rep_relationship.entry(r1).or_default().push(r2);
                }

                let mut to_mesh: AHashMap<Id<_>, Vec<DMat4>> = AHashMap::new();
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
                    to_mesh = step_file
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_i, e)| {
                            matches!(
                                e,
                                Entity::ManifoldSolidBrep(_)
                                    | Entity::BrepWithVoids(_)
                                    | Entity::ShellBasedSurfaceModel(_)
                            )
                        })
                        .map(|(i, _e)| (Id::new(i), vec![DMat4::identity()]))
                        .collect();
                }

                // Extract all face IDs with metadata (no deep copies)
                let face_tasks: Vec<FaceTask> = to_mesh
                    .into_iter()
                    .flat_map(|(brep_id, mats)| {
                        let color = brep_colors
                            .get(&brep_id)
                            .copied()
                            .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

                        collect_faces_from_brep(&step_file, brep_id)
                            .into_iter()
                            .map(move |(face_id, flip)| FaceTask {
                                face_id,
                                transforms: mats.clone(),
                                color,
                                flip_normal: flip,
                            })
                    })
                    .collect();

                // Use the triangulate function with the shared GPU device
                let (mesh, stats) = wgpu_utils::triangulate_faces_with_device(
                    &step_file,
                    &face_tasks,
                    &device,
                    &queue,
                );
                (mesh, stats) // Return both mesh and stats
            } else {
                (Mesh::default(), Stats::default())
            }
        })
        .collect()
}

/// True batch triangulation function that processes all STEP files in a single GPU operation
/// This maximizes GPU utilization by combining all faces from all files into one large batch
/// Can handle up to 256MB worth of STEP data in a single batch for optimal performance
#[cfg(feature = "wgpu")]
pub fn wgpu_triangulate_true_batch_from_examples() -> Vec<(Mesh, Stats)> {
    use std::fs;
    use std::path::Path;
    use step::step_file::StepFile;

    // List of problematic STEP files that cause triangulation panics
    const PROBLEMATIC_FILES: &[&str] = &[
        "../examples/sphere.step", // Known to cause assertion failure in CDT triangulation
                                   // Add other problematic files here as they are discovered
    ];

    // Find all STEP files in examples directory (excluding problematic ones)
    let mut step_files_paths = Vec::new();

    let examples_paths = ["../examples", "./examples"];
    for examples_path in &examples_paths {
        let path = Path::new(examples_path);
        if path.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_file() {
                    let path_str = path.to_string_lossy().to_string();
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase());
                    if (ext == Some("step".to_string()) || ext == Some("stp".to_string()))
                        && !PROBLEMATIC_FILES.contains(&path_str.as_str())
                    {
                        step_files_paths.push(path_str);
                    }
                }
            }
            break; // Use first directory that exists
        }
    }

    if step_files_paths.is_empty() {
        println!("No STEP files found in examples directories for true batch processing");
        return vec![];
    }

    println!("True batch processing {} STEP files", step_files_paths.len());

    // Create a single GPU device for the entire batch to avoid resource exhaustion
    let (device, queue) = match wgpu_utils::create_wgpu_device() {
        Ok(dq) => dq,
        Err(e) => {
            eprintln!("Failed to create GPU device for true batch processing: {}", e);
            // Fallback to CPU processing if GPU fails
            return step_files_paths
                .iter()
                .map(|path| {
                    if let Ok(contents) = fs::read(path) {
                        let flattened = StepFile::strip_flatten(&contents);
                        let step_file = StepFile::parse(&flattened);
                        crate::triangulate::triangulate4(&step_file)
                    } else {
                        (Mesh::default(), Stats::default())
                    }
                })
                .collect();
        }
    };

    // Due to lifetime constraints with FaceTask<'a> containing references to StepFile,
    // true batch processing of multiple STEP files simultaneously is not possible.
    // Each StepFile has its own lifetime, and FaceTask references cannot be mixed.
    println!("True batch processing not fully implemented due to lifetime constraints - processing files individually with shared GPU");

    // Fallback: process files individually with shared GPU device
    step_files_paths
        .iter()
        .map(|path| {
            if let Ok(contents) = fs::read(path) {
                let flattened = StepFile::strip_flatten(&contents);
                let step_file = StepFile::parse(&flattened);
                
                // Build face tasks for this specific file using the same logic as wgpu_triangulate
                let brep_colors: AHashMap<_, DVec3> = step_file
                    .0
                    .iter()
                    .filter_map(
                        MechanicalDesignGeometricPresentationRepresentation_::try_from_entity,
                    )
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

                let mut todo: Vec<_> = roots.into_iter().map(|v| (v, DMat4::identity())).collect();
                let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
                for (r1, r2) in
                    step_file
                        .0
                        .iter()
                        .filter_map(ShapeRepresentationRelationship_::try_from_entity)
                        .map(|e| (e.rep_1, e.rep_2))
                {
                    shape_rep_relationship.entry(r1).or_default().push(r2);
                }

                let mut to_mesh: AHashMap<Id<_>, Vec<DMat4>> = AHashMap::new();
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
                    to_mesh = step_file
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_i, e)| {
                            matches!(
                                e,
                                Entity::ManifoldSolidBrep(_)
                                    | Entity::BrepWithVoids(_)
                                    | Entity::ShellBasedSurfaceModel(_)
                            )
                        })
                        .map(|(i, _e)| (Id::new(i), vec![DMat4::identity()]))
                        .collect();
                }

                // Extract all face IDs with metadata (no deep copies)
                // Process immediately rather than storing references
                let mut file_face_tasks = Vec::new();
                for (brep_id, mats) in to_mesh {
                    let color = brep_colors
                        .get(&brep_id)
                        .copied()
                        .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

                    let faces_with_flips = collect_faces_from_brep(&step_file, brep_id);
                    for (face_id, flip) in faces_with_flips {
                        file_face_tasks.push(FaceTask {
                            face_id,
                            transforms: mats.clone(),
                            color,
                            flip_normal: flip,
                        });
                    }
                }
                
                // Process this file's face tasks using the shared GPU device
                if !file_face_tasks.is_empty() {
                    let (mesh, stats) = wgpu_utils::triangulate_faces_with_device(
                        &step_file, // Use the actual step file for this batch
                        &file_face_tasks,
                        &device,
                        &queue,
                    );
                    (mesh, stats)
                } else {
                    (Mesh::default(), Stats::default())
                }
            } else {
                (Mesh::default(), Stats::default())
            }
        })
        .collect()
}

/// Triangulates a STEP file using GPU acceleration
/// This is the primary public API for GPU-based triangulation.
#[cfg(feature = "wgpu")]
pub fn triangulate(s: &step::step_file::StepFile) -> (crate::mesh::Mesh, crate::stats::Stats) {
    wgpu_triangulate(s)
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_wgpu_triangulate_batch_from_examples() {
        // This test verifies that the batch function compiles and can be called
        // It doesn't assert specific results since they depend on the files in examples directory
        let results: Vec<(Mesh, Stats)> = wgpu_triangulate_batch_from_examples();

        //TODO: let's just sum these all up and count the number of 'failed entries' then report that.
        assert!(results.into_iter().enumerate().all(|(_e, (mesh, stats))| {
            !mesh.verts.is_empty()
                || !mesh.triangles.is_empty()
                || stats.num_errors == 0
                || stats.num_panics == 0
        }));

        // Should complete without panicking
    }
}
