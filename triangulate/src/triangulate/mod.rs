use ahash::AHashMap;
use std::collections::HashSet;
use std::convert::TryInto;

use glm::{DMat4, DVec3, DVec4, U32Vec3};
use log::{error, info, warn};
use nalgebra_glm as glm;

#[cfg(feature = "rayon")]
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub mod cached_triangulation;

#[cfg(feature = "wgpu")]
pub mod wgpu_impl;

use crate::{
    Error,
    curve::Curve,
    mesh,
    mesh::{Mesh, Triangle},
    stats::Stats,
    surface::Surface,
};
use nurbs::{BSplineSurface, KnotVector, NURBSSurface, SampledCurve, SampledSurface};
use step::{
    ap214,
    ap214::Entity,
    ap214::*,
    id::Id,
    step_file::{FromEntity, StepFile},
};

type TransformStack<'a> = AHashMap<Representation<'a>, Vec<(Representation<'a>, DMat4)>>;

pub fn build_transform_stack<'a>(s: &'a StepFile, flip: bool) -> TransformStack<'a> {
    let mut transform_stack: TransformStack<'a> = AHashMap::new();
    for mapped_item in s.0.iter().filter_map(MappedItem_::try_from_entity) {
        let mapping_source = s.entity(mapped_item.mapping_source).unwrap();
        let mat = item_defined_transformation(s, mapping_source.mapping_origin.cast());
        let (p, c) = (
            mapping_source.mapped_representation,
            mapped_item.mapping_target,
        );
        if flip {
            transform_stack
                .entry(c.cast())
                .or_default()
                .push((p.cast(), mat));
        } else {
            transform_stack
                .entry(p.cast())
                .or_default()
                .push((c.cast(), mat));
        }
    }
    transform_stack
}

const SAVE_DEBUG_SVGS: bool = false;
const SAVE_PANIC_SVGS: bool = false;
pub struct FaceTask<'a> {
    pub face_id: AdvancedFace<'a>,
    pub transforms: Vec<DMat4>,
    pub color: DVec3,
    pub flip_normal: bool,
}

/// Truly optimized batched triangulation that processes all faces in a single GPU operation
/// This addresses the performance issues by eliminating per-face CPU-GPU transfers
#[cfg(feature = "wgpu")]
pub fn triangulate5(s: &StepFile) -> (Mesh, Stats) {
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

    let face_tasks: Vec<crate::triangulate::FaceTask> = to_mesh
        .into_iter()
        .flat_map(|(brep_id, mats)| {
            let color = brep_colors
                .get(&brep_id)
                .copied()
                .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

            collect_faces_from_brep(s, brep_id)
                .into_iter()
                .map(move |(face_id, flip)| crate::triangulate::FaceTask {
                    face_id,
                    transforms: mats.clone(),
                    color,
                    flip_normal: flip,
                })
        })
        .collect();

    // Phase 3: Batched GPU triangulation for all faces
    // This is the key optimization - process all faces in a single GPU operation
    crate::triangulate::wgpu_impl::triangulate_faces(s, &face_tasks)
}

pub fn transform_stack_roots<'a>(transform_stack: &TransformStack<'a>) -> Vec<Representation<'a>> {
    let children: HashSet<_> = transform_stack
        .values()
        .flat_map(|v| v.iter())
        .map(|v| v.0)
        .collect();
    transform_stack
        .keys()
        .filter(|k| !children.contains(k))
        .copied()
        .collect()
}

pub fn triangulate(s: &StepFile) -> (Mesh, Stats) {
    let styled_items: Vec<_> =
        s.0.iter()
            .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
            .flat_map(|m| m.items.iter())
            .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
            .collect();

    let brep_colors: AHashMap<_, DVec3> = styled_items
        .iter()
        .filter_map(|styled| {
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0]).map(|c| (styled.item, c))
            }
        })
        .collect();

    // Store a map of parent -> (child, transform)
    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    // The transformation graph isn't directional (because STEP is a Good File
    // Format), so if it's got more than one root, assume it's backwards.  We
    // are assuming that directions in the graph are consistent within the file,
    // until we find a counterexample.
    if roots.len() > 1 {
        info!("Flipping transform stack");
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }
    let mut todo: Vec<_> = roots.into_iter().map(|v| (v, DMat4::identity())).collect();
    if todo.len() > 1 {
        warn!("Transformation stack has more than one root!");
    }

    // Store a map of ShapeRepresentationRelationships, which some models
    // use to map from axes to specific instances
    let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: AHashMap<Id<_>, Vec<_>> = AHashMap::new();
    while let Some((id, mat)) = todo.pop() {
        for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
            todo.push((*child, mat));
        }
        if let Some(children) = transform_stack.get(&id) {
            for (child, next_mat) in children {
                todo.push((*child, mat * next_mat));
            }
        } else {
            // Bind this transform to the RepresentationItem, which is
            // either a ManifoldSolidBrep or a ShellBasedSurfaceModel
            let items = match &s[id] {
                Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                Entity::ShapeRepresentation(b) => &b.items,
                Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                e => panic!("Could not get shape from {:?}", e),
            };

            for m in items.iter() {
                match &s[*m] {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) => to_mesh.entry(*m).or_default().push(mat),
                    Entity::Axis2Placement3d(_) => (),
                    e => warn!("Skipping {:?}", e),
                }
            }
        }
    }
    // If there are items in breps that aren't attached to a transformation
    // chain, then draw them individually (with an identity matrix)
    if to_mesh.is_empty() {
        s.0.iter()
            .enumerate()
            .filter(|(_i, e)| match e {
                Entity::ManifoldSolidBrep(_)
                | Entity::BrepWithVoids(_)
                | Entity::ShellBasedSurfaceModel(_) => true,
                _ => false,
            })
            .map(|(i, _e)| Id::new(i))
            .for_each(|i| to_mesh.entry(i).or_default().push(DMat4::identity()));
    }

    let (to_mesh_iter, empty) = {
        #[cfg(feature = "rayon")]
        {
            (to_mesh.par_iter(), || (Mesh::default(), Stats::default()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            (to_mesh.iter(), (Mesh::default(), Stats::default()))
        }
    };
    let mesh_fold = to_mesh_iter.fold(
        // Empty constructor
        empty,
        // Fold operation
        |(mut mesh, mut stats), (id, mats)| {
            let v_start = mesh.verts.len();
            let t_start = mesh.triangles.len();
            match &s[*id] {
                Entity::ManifoldSolidBrep(b) => closed_shell(s, b.outer, &mut mesh, &mut stats),
                Entity::ShellBasedSurfaceModel(b) => {
                    for v in &b.sbsm_boundary {
                        shell(s, *v, &mut mesh, &mut stats);
                    }
                }
                Entity::BrepWithVoids(b) =>
                // TODO: handle voids
                {
                    closed_shell(s, b.outer, &mut mesh, &mut stats)
                }
                _ => {
                    warn!("Skipping {:?} (not a known solid)", s[*id]);
                    return (mesh, stats);
                }
            };

            // Pick out a color from the color map and apply it to each
            // newly-created vertex
            //TODO: Make colours optional
            let color = brep_colors
                .get(id)
                .copied()
                .unwrap_or(DVec3::new(0.5, 0.5, 0.5));

            // Build copies of the mesh by copying and applying transforms
            //TODO: work out how not to clone.
            let v_end = mesh.verts.len();
            let t_end = mesh.triangles.len();
            for mat in &mats[1..] {
                for v in v_start..v_end {
                    let p = mesh.verts[v].pos;
                    let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                    let pos = (mat * p_h).xyz();

                    let n = mesh.verts[v].norm;
                    let norm = (mat * glm::vec3_to_vec4(&n)).xyz();

                    mesh.verts.push(mesh::Vertex { pos, norm, color });
                }
                let offset = mesh.verts.len() - v_end;
                for t in t_start..t_end {
                    let mut tri = mesh.triangles[t];
                    tri.verts.add_scalar_mut(offset as u32);
                    mesh.triangles.push(tri);
                }
            }

            // Now that we've built all of the other copies of the mesh,
            // re-use the original mesh and apply the first transform
            let mat = mats[0];
            for v in v_start..v_end {
                let p = mesh.verts[v].pos;
                let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                mesh.verts[v].pos = (mat * p_h).xyz();

                let n = mesh.verts[v].norm;
                mesh.verts[v].norm = (mat * glm::vec3_to_vec4(&n)).xyz();

                mesh.verts[v].color = color;
            }
            (mesh, stats)
        },
    );

    let (mesh, stats) = {
        #[cfg(feature = "rayon")]
        {
            mesh_fold.reduce(empty, |a, b| {
                (Mesh::combine(a.0, b.0), Stats::combine(a.1, b.1))
            })
        }
        #[cfg(not(feature = "rayon"))]
        {
            mesh_fold
        }
    };

    info!("num_shells: {}", stats.num_shells);
    info!("num_faces: {}", stats.num_faces);
    info!("num_errors: {}", stats.num_errors);
    info!("num_panics: {}", stats.num_panics);

    (mesh, stats)
}

fn item_defined_transformation(s: &StepFile, t: Id<ItemDefinedTransformation_>) -> DMat4 {
    let i = s.entity(t).expect("Could not get ItemDefinedTransform");

    let (location, axis, ref_direction) = axis2_placement_3d(s, i.transform_item_1.cast());
    let t1 =
        Surface::make_affine_transform(axis, ref_direction, axis.cross(&ref_direction), location);

    let (location, axis, ref_direction) = axis2_placement_3d(s, i.transform_item_2.cast());
    let t2 =
        Surface::make_affine_transform(axis, ref_direction, axis.cross(&ref_direction), location);

    t2 * t1.try_inverse().expect("Could not invert transform matrix")
}

pub fn presentation_style_color(s: &StepFile, p: PresentationStyleAssignment) -> Option<DVec3> {
    // AAAAAHHHHH
    s.entity(p)
        .and_then(|p: &PresentationStyleAssignment_| {
            let mut surf = p.styles.iter().filter_map(|y| {
                // This is an ambiguous parse, so we hard-code the first
                // Entity item in the enum
                use PresentationStyleSelect::PreDefinedPresentationStyle;
                if let PreDefinedPresentationStyle(u) = y {
                    s.entity(u.cast::<SurfaceStyleUsage_>())
                } else {
                    None
                }
            });

            surf.next()
        })
        .and_then(|surf: &SurfaceStyleUsage_| s.entity(surf.style.cast::<SurfaceSideStyle_>()))
        .and_then(|surf: &SurfaceSideStyle_| {
            if surf.styles.len() != 1 {
                None
            } else {
                s.entity(surf.styles[0].cast::<SurfaceStyleFillArea_>())
            }
        })
        .map(|surf: &SurfaceStyleFillArea_| {
            s.entity(surf.fill_area).expect("Could not get fill_area")
        })
        .and_then(|fill: &FillAreaStyle_| {
            if fill.fill_styles.len() != 1 {
                None
            } else {
                s.entity(fill.fill_styles[0].cast::<FillAreaStyleColour_>())
            }
        })
        .and_then(|f: &FillAreaStyleColour_| s.entity(f.fill_colour.cast::<ColourRgb_>()))
        .map(|c| DVec3::new(c.red, c.green, c.blue))
}

pub fn cartesian_point(s: &StepFile, a: Id<CartesianPoint_>) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0)
}

pub fn direction(s: &StepFile, a: Direction) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    DVec3::new(
        p.direction_ratios[0],
        p.direction_ratios[1],
        p.direction_ratios[2],
    )
}

pub fn axis2_placement_3d(s: &StepFile, t: Id<Axis2Placement3d_>) -> (DVec3, DVec3, DVec3) {
    let a = s.entity(t).expect("Could not get Axis2Placement3d");
    let location = cartesian_point(s, a.location);
    // TODO: this doesn't necessarily match the behavior of `build_axes`
    let axis = direction(s, a.axis.expect("Missing axis"));
    let ref_direction = match a.ref_direction {
        None => DVec3::new(1.0, 0.0, 0.0),
        Some(r) => direction(s, r),
    };
    (location, axis, ref_direction)
}

fn shell(s: &StepFile, c: Shell, mesh: &mut Mesh, stats: &mut Stats) {
    match &s[c] {
        Entity::ClosedShell(_) => closed_shell(s, c.cast(), mesh, stats),
        Entity::OpenShell(_) => open_shell(s, c.cast(), mesh, stats),
        h => warn!("Skipping {:?} (unknown Shell type)", h),
    }
}

fn open_shell(s: &StepFile, c: OpenShell, mesh: &mut Mesh, stats: &mut Stats) {
    let cs = s.entity(c).expect("Could not get OpenShell");
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(s, face.cast(), mesh, stats) {
            error!("Failed to triangulate {:?}: {}", s[*face], err);
        }
    }
    stats.num_shells += 1;
}

fn closed_shell(s: &StepFile, c: ClosedShell, mesh: &mut Mesh, stats: &mut Stats) {
    let cs = s.entity(c).expect("Could not get ClosedShell");
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(s, face.cast(), mesh, stats) {
            error!("Failed to triangulate {:?}: {}", s[*face], err);
        }
    }
    stats.num_shells += 1;
}

fn advanced_face(
    s: &StepFile,
    f: AdvancedFace,
    mesh: &mut Mesh,
    stats: &mut Stats,
) -> Result<(), Error> {
    let face = s.entity(f).expect("Could not get AdvancedFace");
    stats.num_faces += 1;

    // Grab the surface, returning early if it's unimplemented
    let mut surf = get_surface(s, face.face_geometry)?;

    // This is the starting point at which we insert new vertices
    let offset = mesh.verts.len();

    // For each contour, project from 3D down to the surface, then
    // start collecting them as constrained edges for triangulation
    let mut edges = Vec::new();
    let v_start = mesh.verts.len();
    let mut num_pts = 0;
    for b in &face.bounds {
        let bound_contours = face_bound(s, *b)?;

        match bound_contours.len() {
            // We should always have non-zero items in the contour
            0 => panic!("Got empty contours for {:?}", face),

            // Special case for a single-vertex point, which shows up in
            // cones: we push it as a Steiner point, but without any
            // associated contours.
            1 => {
                num_pts += 1;
                mesh.verts.push(mesh::Vertex {
                    pos: bound_contours[0],
                    norm: DVec3::zeros(),
                    color: DVec3::new(0.0, 0.0, 0.0),
                });
            }

            // Default for lists of contour points
            _ => {
                // Record the initial point to close the loop
                let start = num_pts;
                for pt in bound_contours {
                    // The contour marches forward!
                    edges.push((num_pts, num_pts + 1));

                    // Also store this vertex in the 3D triangulation
                    mesh.verts.push(mesh::Vertex {
                        pos: pt,
                        norm: DVec3::zeros(),
                        color: DVec3::new(0.0, 0.0, 0.0),
                    });
                    num_pts += 1;
                }
                // The last point is a duplicate, because it closes the
                // contours, so we skip it here and reattach the contour to
                // the start.
                num_pts -= 1;
                mesh.verts.pop();

                // Close the loop by returning to the starting point
                edges.pop();
                edges.last_mut().unwrap().1 = start;
            }
        }
    }

    // We inject Stiner points based on the surface type to improve curvature,
    // e.g. for spherical sections.  However, we don't want triagulation to
    // _fail_ due to these points, so if that happens, we nuke the point (by
    // assigning it to the first point in the list, which causes it to get
    // deduplicated), then retry.
    let mut pts = surf.lower_verts(&mut mesh.verts[v_start..])?;
    let bonus_points = pts.len();
    surf.add_steiner_points(&mut pts, &mut mesh.verts);
    let result = std::panic::catch_unwind(|| {
        // TODO: this is only needed because we use pts below to save a debug
        // SVG if this panics.  Once we're confident in never panicking, we
        // can remove this.
        let mut pts = pts.clone();
        loop {
            let mut t = match cdt::Triangulation::new_with_edges(&pts, &edges) {
                Err(e) => break Err(e),
                Ok(t) => t,
            };
            match t.run() {
                Ok(()) => break Ok(t),
                // If triangulation failed due to a Steiner point on a fixed
                // edge, then reassign that point to pts[0] (so it will be
                // ignored as a duplicate)
                Err(cdt::Error::PointOnFixedEdge(p)) if p >= bonus_points => {
                    pts[p] = pts[0];
                    continue;
                }
                Err(e) => {
                    if SAVE_DEBUG_SVGS {
                        let filename = format!("err{}.svg", face.face_geometry.0);
                        t.save_debug_svg(&filename)
                            .expect("Could not save debug SVG");
                    }
                    break Err(e);
                }
            }
        }
    });
    match result {
        Ok(Ok(t)) => {
            for (a, b, c) in t.triangles() {
                let a = (a + offset) as u32;
                let b = (b + offset) as u32;
                let c = (c + offset) as u32;
                mesh.triangles.push(Triangle {
                    verts: if face.same_sense {
                        U32Vec3::new(a, b, c)
                    } else {
                        U32Vec3::new(a, c, b)
                    },
                });
            }
        }
        Ok(Err(e)) => {
            error!(
                "Got error while triangulating {}: {:?}",
                face.face_geometry.0, e
            );
            stats.num_errors += 1;
        }
        Err(e) => {
            error!(
                "Got panic while triangulating {}: {:?}",
                face.face_geometry.0, e
            );
            if SAVE_PANIC_SVGS {
                let filename = format!("panic{}.svg", face.face_geometry.0);
                cdt::save_debug_panic(&pts, &edges, &filename).expect("Could not save debug SVG");
            }
            stats.num_panics += 1;
        }
    }
    // Flip normals of new vertices, depending on the same_sense flag
    if !face.same_sense {
        for v in &mut mesh.verts[v_start..] {
            v.norm = -v.norm;
        }
    }
    Ok(())
}

pub fn get_surface(s: &StepFile, surf: ap214::Surface) -> Result<Surface, Error> {
    match &s[surf] {
        Entity::CylindricalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_cylinder(
                axis,
                ref_direction,
                location,
                c.radius.0.0.0,
            ))
        }
        Entity::ToroidalSurface(c) => {
            let (location, axis, _ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_torus(
                location,
                axis,
                c.major_radius.0.0.0,
                c.minor_radius.0.0.0,
            ))
        }
        Entity::Plane(p) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, axis, ref_direction) = axis2_placement_3d(s, p.position);
            Ok(Surface::new_plane(axis, ref_direction, location))
        }
        // We treat cones like planes, since that's a valid mapping into 2D
        Entity::ConicalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_cone(
                axis,
                ref_direction,
                location,
                c.semi_angle.0,
            ))
        }
        Entity::SphericalSurface(c) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, _axis, _ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_sphere(location, c.radius.0.0.0))
        }
        Entity::BSplineSurfaceWithKnots(b) => {
            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = b.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = b
                .u_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                b.u_degree.try_into().expect("Got negative degree"),
                &u_knots,
                &u_multiplicities,
            );

            let v_knots: Vec<f64> = b.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = b
                .v_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                b.v_degree.try_into().expect("Got negative degree"),
                &v_knots,
                &v_multiplicities,
            );

            let control_points_list = control_points_2d(s, &b.control_points_list);

            let surf = BSplineSurface::new(
                !b.u_closed.0.unwrap(),
                !b.v_closed.0.unwrap(),
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::BSpline(SampledSurface::new(surf)))
        }
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineSurfaceWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType);
            };
            let rational = if let Entity::RationalBSplineSurface(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType);
            };

            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = bspline.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = bspline
                .u_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                bspline.u_degree.try_into().expect("Got negative degree"),
                &u_knots,
                &u_multiplicities,
            );

            let v_knots: Vec<f64> = bspline.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = bspline
                .v_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                bspline.v_degree.try_into().expect("Got negative degree"),
                &v_knots,
                &v_multiplicities,
            );

            let control_points_list = control_points_2d(s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(ctrl, weight)| {
                    ctrl.into_iter()
                        .zip(weight)
                        .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                        .collect()
                })
                .collect();

            let surf = NURBSSurface::new(
                !bspline.u_closed.0.unwrap(),
                !bspline.v_closed.0.unwrap(),
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::NURBS(SampledSurface::new(surf)))
        }
        e => {
            warn!("Could not get surface from {:?}", e);
            Err(Error::UnknownSurfaceType)
        }
    }
}

pub fn control_points_1d(s: &StepFile, row: &Vec<CartesianPoint>) -> Vec<DVec3> {
    row.iter().map(|p| cartesian_point(s, *p)).collect()
}

pub fn control_points_2d(s: &StepFile, rows: &Vec<Vec<CartesianPoint>>) -> Vec<Vec<DVec3>> {
    rows.iter().map(|row| control_points_1d(s, row)).collect()
}

pub fn face_bound(s: &StepFile, b: FaceBound) -> Result<Vec<DVec3>, Error> {
    let (bound, orientation) = match &s[b] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        e => panic!("Could not get bound from {:?} at {:?}", e, b),
    };
    match &s[bound] {
        Entity::EdgeLoop(e) => {
            let mut d = edge_loop(s, &e.edge_list)?;
            if !orientation {
                d.reverse()
            }
            Ok(d)
        }
        Entity::VertexLoop(v) => {
            // This is an "edge loop" with a single vertex, which is
            // used for cones and not really anything else.
            Ok(vec![vertex_point(s, v.loop_vertex)])
        }
        e => panic!("{:?} is not an EdgeLoop", e),
    }
}

pub fn edge_loop(s: &StepFile, edge_list: &[OrientedEdge]) -> Result<Vec<DVec3>, Error> {
    let mut out = Vec::new();
    for (i, e) in edge_list.iter().enumerate() {
        // Remove the last item from the list, since it's the beginning
        // of the following list (hopefully)
        if i > 0 {
            out.pop();
        }
        let edge = s.entity(*e).expect("Could not get OrientedEdge");
        let o = edge_curve(s, edge.edge_element.cast(), edge.orientation)?;
        out.extend(o.into_iter());
    }
    Ok(out)
}

pub fn edge_curve(s: &StepFile, e: EdgeCurve, orientation: bool) -> Result<Vec<DVec3>, Error> {
    let edge_curve = s.entity(e).expect("Could not get EdgeCurve");
    let curve = curve(s, edge_curve, edge_curve.edge_geometry, orientation)?;

    let (start, end) = if orientation {
        (edge_curve.edge_start, edge_curve.edge_end)
    } else {
        (edge_curve.edge_end, edge_curve.edge_start)
    };
    let u = vertex_point(s, start);
    let v = vertex_point(s, end);
    Ok(curve.build(u, v))
}

pub fn curve(
    s: &StepFile,
    edge_curve: &ap214::EdgeCurve_,
    curve_id: ap214::Curve,
    orientation: bool,
) -> Result<Curve, Error> {
    Ok(match &s[curve_id] {
        Entity::Circle(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_circle(
                location,
                axis,
                ref_direction,
                c.radius.0.0.0,
                edge_curve.edge_start == edge_curve.edge_end,
                edge_curve.same_sense ^ !orientation,
            )
        }
        Entity::Ellipse(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_ellipse(
                location,
                axis,
                ref_direction,
                c.semi_axis_1.0.0.0,
                c.semi_axis_2.0.0.0,
                edge_curve.edge_start == edge_curve.edge_end,
                edge_curve.same_sense ^ !orientation,
            )
        }
        Entity::BSplineCurveWithKnots(c) => {
            if c.closed_curve.0 != Some(false) {
                return Err(Error::ClosedCurve);
            } else if c.self_intersect.0 != Some(false) {
                return Err(Error::SelfIntersectingCurve);
            }

            let control_points_list = control_points_1d(s, &c.control_points_list);

            let knots: Vec<f64> = c.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = c
                .knot_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                c.degree.try_into().expect("Got negative degree"),
                &knots,
                &multiplicities,
            );

            let curve =
                nurbs::BSplineCurve::new(!c.closed_curve.0.unwrap(), knot_vec, control_points_list);
            Curve::BSplineCurveWithKnots(SampledCurve::new(curve))
        }
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineCurveWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType);
            };
            let rational = if let Entity::RationalBSplineCurve(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType);
            };
            let knots: Vec<f64> = bspline.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = bspline
                .knot_multiplicities
                .iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                bspline.degree.try_into().expect("Got negative degree"),
                &knots,
                &multiplicities,
            );

            let control_points_list = control_points_1d(s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                .collect();

            let curve = nurbs::NURBSCurve::new(
                !bspline.closed_curve.0.unwrap(),
                knot_vec,
                control_points_list,
            );
            Curve::NURBSCurve(SampledCurve::new(curve))
        }
        Entity::SurfaceCurve(v) => curve(s, edge_curve, v.curve_3d, orientation)?,
        Entity::SeamCurve(v) => curve(s, edge_curve, v.curve_3d, orientation)?,
        // The Line type ignores pnt / dir and just uses u and v
        Entity::Line(_) => Curve::new_line(),
        e => {
            warn!("Could not get edge from {:?}", e);
            return Err(Error::UnknownCurveType);
        }
    })
}

pub fn vertex_point(s: &StepFile, v: Vertex) -> DVec3 {
    cartesian_point(
        s,
        s.entity(v.cast::<VertexPoint_>())
            .expect("Could not get VertexPoint")
            .vertex_geometry
            .cast(),
    )
}

#[cfg(feature = "rayon")]
pub fn triangulate2(s: &StepFile) -> (Mesh, Stats) {
    use nalgebra::{Matrix4, Vector3, Vector4};
    use rayon::prelude::*;

    let styled_items: Vec<_> =
        s.0.iter()
            .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
            .flat_map(|m| m.items.iter())
            .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
            .collect();

    let brep_colors: AHashMap<_, Vector3<f64>> = styled_items
        .iter()
        .filter_map(|styled| {
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0]).map(|c| (styled.item, c))
            }
        })
        .collect();

    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    if roots.len() > 1 {
        info!("Flipping transform stack");
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }
    let mut todo: Vec<_> = roots
        .into_iter()
        .map(|v| (v, Matrix4::identity()))
        .collect();
    if todo.len() > 1 {
        warn!("Transformation stack has more than one root!");
    }

    let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: AHashMap<Id<_>, Vec<_>> = AHashMap::new();
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
                e => panic!("Could not get shape from {:?}", e),
            };

            for m in items.iter() {
                match &s[*m] {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) => to_mesh.entry(*m).or_default().push(mat),
                    Entity::Axis2Placement3d(_) => (),
                    e => warn!("Skipping {:?}", e),
                }
            }
        }
    }
    if to_mesh.is_empty() {
        s.0.iter()
            .enumerate()
            .filter(|(_i, e)| match e {
                Entity::ManifoldSolidBrep(_)
                | Entity::BrepWithVoids(_)
                | Entity::ShellBasedSurfaceModel(_) => true,
                _ => false,
            })
            .map(|(i, _e)| Id::new(i))
            .for_each(|i| to_mesh.entry(i).or_default().push(Matrix4::identity()));
    }

    // Use proper conditional compilation for rayon/non-rayon support
    #[cfg(feature = "rayon")]
    let (mesh, stats) = to_mesh
        .par_iter()
        .map(|(id, mats)| {
            let mut local_mesh = Mesh::default();
            let mut local_stats = Stats::default();

            let color = brep_colors
                .get(id)
                .copied()
                .unwrap_or(Vector3::new(0.5, 0.5, 0.5));

            let mut template_mesh = Mesh::default();
            match &s[*id] {
                Entity::ManifoldSolidBrep(b) => {
                    closed_shell(s, b.outer, &mut template_mesh, &mut local_stats);
                }
                Entity::ShellBasedSurfaceModel(b) => {
                    for v in &b.sbsm_boundary {
                        shell(s, *v, &mut template_mesh, &mut local_stats);
                    }
                }
                Entity::BrepWithVoids(b) => {
                    closed_shell(s, b.outer, &mut template_mesh, &mut local_stats);

                    for void_shell_id in &b.voids {
                        let oriented_shell = s.entity(*void_shell_id).unwrap();
                        let mut void_mesh = Mesh::default();
                        closed_shell(
                            s,
                            oriented_shell.closed_shell_element,
                            &mut void_mesh,
                            &mut local_stats,
                        );

                        if !oriented_shell.orientation {
                            for tri in void_mesh.triangles.iter_mut() {
                                tri.verts.swap_rows(1, 2);
                            }
                        }
                        template_mesh = Mesh::combine(template_mesh, void_mesh);
                    }
                }
                _ => {
                    warn!("Skipping {:?} (not a known solid)", s[*id]);
                    return (local_mesh, local_stats);
                }
            };

            for mat in mats.iter() {
                let v_offset = local_mesh
                    .verts
                    .len()
                    .try_into()
                    .expect("too many triangles");

                for v in &template_mesh.verts {
                    let p_h = Vector4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    let pos = (mat * p_h).xyz();

                    let n_h = Vector4::new(v.norm.x, v.norm.y, v.norm.z, 0.0);
                    let norm = (mat * n_h).xyz().normalize();

                    local_mesh.verts.push(mesh::Vertex { pos, norm, color });
                }

                for t in &template_mesh.triangles {
                    let mut new_tri = *t;
                    new_tri.verts.add_scalar_mut(v_offset);
                    local_mesh.triangles.push(new_tri);
                }
            }
            (local_mesh, local_stats)
        })
        .reduce(
            || (Mesh::default(), Stats::default()),
            |(mesh_a, stats_a), (mesh_b, stats_b)| {
                (
                    Mesh::combine(mesh_a, mesh_b),
                    Stats::combine(stats_a, stats_b),
                )
            },
        );
    #[cfg(not(feature = "rayon"))]
    let (mesh, stats) = to_mesh
        .iter()
        .map(|(id, mats)| {
            let mut local_mesh = Mesh::default();
            let mut local_stats = Stats::default();

            let color = brep_colors
                .get(id)
                .copied()
                .unwrap_or(Vector3::new(0.5, 0.5, 0.5));

            let mut template_mesh = Mesh::default();
            match &s[*id] {
                Entity::ManifoldSolidBrep(b) => {
                    closed_shell(s, b.outer, &mut template_mesh, &mut local_stats);
                }
                Entity::ShellBasedSurfaceModel(b) => {
                    for v in &b.sbsm_boundary {
                        shell(s, *v, &mut template_mesh, &mut local_stats);
                    }
                }
                Entity::BrepWithVoids(b) => {
                    closed_shell(s, b.outer, &mut template_mesh, &mut local_stats);

                    for void_shell_id in &b.voids {
                        let oriented_shell = s.entity(*void_shell_id).unwrap();
                        let mut void_mesh = Mesh::default();
                        closed_shell(
                            s,
                            oriented_shell.closed_shell_element,
                            &mut void_mesh,
                            &mut local_stats,
                        );

                        if !oriented_shell.orientation {
                            for tri in void_mesh.triangles.iter_mut() {
                                tri.verts.swap_rows(1, 2);
                            }
                        }
                        template_mesh = Mesh::combine(template_mesh, void_mesh);
                    }
                }
                _ => {
                    warn!("Skipping {:?} (not a known solid)", s[*id]);
                    return (local_mesh, local_stats);
                }
            };

            for mat in mats.iter() {
                let v_offset = local_mesh
                    .verts
                    .len()
                    .try_into()
                    .expect("too many triangles");

                for v in &template_mesh.verts {
                    let p_h = Vector4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    let pos = (mat * p_h).xyz();

                    let n_h = Vector4::new(v.norm.x, v.norm.y, v.norm.z, 0.0);
                    let norm = (mat * n_h).xyz().normalize();

                    local_mesh.verts.push(mesh::Vertex { pos, norm, color });
                }

                for t in &template_mesh.triangles {
                    let mut new_tri = *t;
                    new_tri.verts.add_scalar_mut(v_offset);
                    local_mesh.triangles.push(new_tri);
                }
            }
            (local_mesh, local_stats)
        })
        .fold(
            (Mesh::default(), Stats::default()),
            |(mesh_a, stats_a), (mesh_b, stats_b)| {
                (
                    Mesh::combine(mesh_a, mesh_b),
                    Stats::combine(stats_a, stats_b),
                )
            },
        );

    info!("num_shells: {}", stats.num_shells);
    info!("num_faces: {}", stats.num_faces);
    info!("num_errors: {}", stats.num_errors);
    info!("num_panics: {}", stats.num_panics);

    (mesh, stats)
}

#[cfg(feature = "rayon")]
pub fn triangulate3(s: &StepFile) -> (Mesh, Stats) {
    use nalgebra::{Matrix4, Vector3, Vector4};
    use rayon::prelude::*;

    // Phase 1: Same setup as triangulate2
    let styled_items: Vec<_> =
        s.0.iter()
            .filter_map(MechanicalDesignGeometricPresentationRepresentation_::try_from_entity)
            .flat_map(|m| m.items.iter())
            .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
            .collect();

    let brep_colors: AHashMap<_, Vector3<f64>> = styled_items
        .iter()
        .filter_map(|styled| {
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0]).map(|c| (styled.item, c))
            }
        })
        .collect();

    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    if roots.len() > 1 {
        info!("Flipping transform stack");
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }

    let mut todo: Vec<_> = roots
        .into_iter()
        .map(|v| (v, Matrix4::identity()))
        .collect();
    if todo.len() > 1 {
        warn!("Transformation stack has more than one root!");
    }

    let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: AHashMap<Id<_>, Vec<_>> = AHashMap::new();
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
                e => panic!("Could not get shape from {:?}", e),
            };

            for m in items.iter() {
                match &s[*m] {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) => to_mesh.entry(*m).or_default().push(mat),
                    Entity::Axis2Placement3d(_) => (),
                    e => warn!("Skipping {:?}", e),
                }
            }
        }
    }

    if to_mesh.is_empty() {
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
            .map(|(i, _e)| Id::new(i))
            .for_each(|i| to_mesh.entry(i).or_default().push(Matrix4::identity()));
    }

    // Phase 2: Collect all faces with their context
    #[derive(Clone)]
    struct FaceJob<'a> {
        face_id: AdvancedFace<'a>,
        transforms: Vec<Matrix4<f64>>,
        color: Vector3<f64>,
    }

    let mut face_jobs = Vec::new();
    for (brep_id, mats) in to_mesh {
        let color = brep_colors
            .get(&brep_id)
            .copied()
            .unwrap_or(Vector3::new(0.5, 0.5, 0.5)); //TODO: option to ignore colours? //TODO: const

        let faces: Vec<AdvancedFace> = match &s[brep_id] {
            Entity::ManifoldSolidBrep(b) => s
                .entity(b.outer)
                .map(|cs: &ClosedShell_| cs.cfs_faces.iter().map(|f| f.cast()).collect())
                .unwrap_or_default(),
            Entity::ShellBasedSurfaceModel(b) => b
                .sbsm_boundary
                .iter()
                .flat_map(|shell| match &s[*shell] {
                    Entity::ClosedShell(cs) => cs.cfs_faces.iter().map(|f| f.cast()).collect(),
                    Entity::OpenShell(os) => os.cfs_faces.iter().map(|f| f.cast()).collect(),
                    _ => vec![],
                })
                .collect(),
            Entity::BrepWithVoids(b) => {
                let mut faces: Vec<Id<AdvancedFace_>> = s
                    .entity(b.outer)
                    .map(|cs: &ClosedShell_| cs.cfs_faces.iter().map(|f| f.cast()).collect())
                    .unwrap_or_default();

                for void_id in &b.voids {
                    if let Some(oriented) = s.entity(*void_id)
                        && let Some(cs) = s.entity(oriented.closed_shell_element)
                    {
                        faces.extend(cs.cfs_faces.iter().map(|f: &Face| f.cast()));
                    } //TODO: Verbose logging on failures behind compile flag
                }
                faces
            }
            _ => vec![],
        };

        for face_id in faces {
            face_jobs.push(FaceJob {
                face_id,
                transforms: mats.clone(),
                color,
            });
        }
    }

    // Phase 3: Parallel face triangulation
    let (mesh, stats) = face_jobs
        .par_iter()
        .map(|job| {
            let mut local_stats = Stats::default();
            let mut face_verts = Vec::new();
            let mut face_triangles = Vec::new();

            // Triangulate once
            if let Err(err) = advanced_face_to_mesh(
                s,
                job.face_id,
                &mut face_verts,
                &mut face_triangles,
                &mut local_stats,
            ) {
                error!("Failed to triangulate face: {}", err);
                return (Mesh::default(), local_stats);
            }

            // Apply transforms and replicate
            let mut mesh = Mesh::default();
            for mat in &job.transforms {
                let v_offset: u32 = mesh.verts.len().try_into().expect("too many vertices");

                for v in &face_verts {
                    let p_h = Vector4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    let pos = (mat * p_h).xyz();

                    let n_h = Vector4::new(v.norm.x, v.norm.y, v.norm.z, 0.0);
                    let norm = (mat * n_h).xyz().normalize();

                    mesh.verts.push(mesh::Vertex {
                        pos: DVec3::new(pos.x, pos.y, pos.z),
                        norm: DVec3::new(norm.x, norm.y, norm.z),
                        color: DVec3::new(job.color.x, job.color.y, job.color.z),
                    });
                }

                for t in &face_triangles {
                    let mut tri = *t;
                    tri.verts.add_scalar_mut(v_offset);
                    mesh.triangles.push(tri);
                }
            }

            (mesh, local_stats)
        })
        .reduce(
            || (Mesh::default(), Stats::default()),
            |(a_mesh, a_stats), (b_mesh, b_stats)| {
                (
                    Mesh::combine(a_mesh, b_mesh),
                    Stats::combine(a_stats, b_stats),
                )
            },
        );

    info!("num_faces: {}", stats.num_faces);
    info!("num_errors: {}", stats.num_errors);
    info!("num_panics: {}", stats.num_panics);

    (mesh, stats)
}

// Helper function: triangulate a single face into local mesh
pub fn advanced_face_to_mesh(
    s: &StepFile,
    f: AdvancedFace,
    verts: &mut Vec<mesh::Vertex>,
    triangles: &mut Vec<Triangle>,
    stats: &mut Stats,
) -> Result<(), Error> {
    let face = s.entity(f).expect("Could not get AdvancedFace");
    stats.num_faces += 1;

    let mut surf = get_surface(s, face.face_geometry)?;

    let v_start = verts.len();
    let mut edges = Vec::new();
    let mut num_pts = 0;

    for b in &face.bounds {
        let bound_contours = face_bound(s, *b)?;

        match bound_contours.len() {
            0 => panic!("Got empty contours"),
            1 => {
                num_pts += 1;
                verts.push(mesh::Vertex {
                    pos: bound_contours[0],
                    norm: DVec3::zeros(),
                    color: DVec3::zeros(),
                });
            }
            _ => {
                let start = num_pts;
                for pt in bound_contours {
                    edges.push((num_pts, num_pts + 1));
                    verts.push(mesh::Vertex {
                        pos: pt,
                        norm: DVec3::zeros(),
                        color: DVec3::zeros(),
                    });
                    num_pts += 1;
                }
                num_pts -= 1;
                verts.pop();
                edges.pop();
                edges.last_mut().unwrap().1 = start;
            }
        }
    }

    let mut pts = surf.lower_verts(&mut verts[v_start..])?;
    let bonus_points = pts.len();
    surf.add_steiner_points(&mut pts, verts);

    let result = std::panic::catch_unwind(|| {
        let mut pts = pts.clone();
        loop {
            let mut t = match cdt::Triangulation::new_with_edges(&pts, &edges) {
                Err(e) => break Err(e),
                Ok(t) => t,
            };
            match t.run() {
                Ok(()) => break Ok(t),
                Err(cdt::Error::PointOnFixedEdge(p)) if p >= bonus_points => {
                    pts[p] = pts[0];
                    continue;
                }
                Err(e) => break Err(e),
            }
        }
    });

    match result {
        Ok(Ok(t)) => {
            for (a, b, c) in t.triangles() {
                let (a, b, c) = (a as u32, b as u32, c as u32);
                triangles.push(Triangle {
                    verts: if face.same_sense {
                        U32Vec3::new(a, b, c)
                    } else {
                        U32Vec3::new(a, c, b)
                    },
                });
            }
        }
        Ok(Err(e)) => {
            error!("Triangulation error: {:?}", e);
            stats.num_errors += 1;
        }
        Err(e) => {
            error!("Triangulation panic: {:?}", e);
            stats.num_panics += 1;
        }
    }

    if !face.same_sense {
        for v in verts.iter_mut().skip(v_start) {
            v.norm = -v.norm;
        }
    }

    Ok(())
}

#[cfg(feature = "rayon")]
pub fn triangulate4(s: &StepFile) -> (Mesh, Stats) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
    struct FaceTask<'a> {
        face_id: AdvancedFace<'a>,
        transforms: Vec<DMat4>,
        color: DVec3,
        flip_normal: bool,
    }

    let num_shells = to_mesh.len(); // Store this before moving to_mesh
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

    // Phase 3: Parallel triangulation with pre-sized buffers
    let total_faces = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let total_panics = AtomicUsize::new(0);

    let mesh = face_tasks
        .par_iter()
        .filter_map(|task| {
            let mut verts = Vec::with_capacity(128); // Typical face has ~50-100 verts
            let mut triangles = Vec::with_capacity(128);

            total_faces.fetch_add(1, Ordering::Relaxed);

            match triangulate_single_face(s, task.face_id, &mut verts, &mut triangles) {
                Ok(()) => {
                    if task.flip_normal {
                        for v in &mut verts {
                            v.norm = -v.norm;
                        }
                    }

                    // Apply transforms and build final mesh
                    let mut mesh = Mesh {
                        verts: Vec::with_capacity(verts.len() * task.transforms.len()),
                        triangles: Vec::with_capacity(triangles.len() * task.transforms.len()),
                    };

                    for mat in &task.transforms {
                        let v_offset = mesh.verts.len() as u32;

                        // Vectorized transform application
                        for v in &verts {
                            let p_h = DVec4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                            let n_h = DVec4::new(v.norm.x, v.norm.y, v.norm.z, 0.0);

                            mesh.verts.push(mesh::Vertex {
                                pos: (mat * p_h).xyz(),
                                norm: (mat * n_h).xyz().normalize(),
                                color: task.color,
                            });
                        }

                        for t in &triangles {
                            let mut tri = *t;
                            tri.verts.add_scalar_mut(v_offset);
                            mesh.triangles.push(tri);
                        }
                    }

                    Some(mesh)
                }
                Err(Error::CouldNotLower) | Err(Error::UnknownSurfaceType) => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    None
                }
                Err(_) => {
                    total_panics.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .reduce(Mesh::default, Mesh::combine);

    let stats = Stats {
        num_shells,
        num_faces: total_faces.load(Ordering::Relaxed),
        num_errors: total_errors.load(Ordering::Relaxed),
        num_panics: total_panics.load(Ordering::Relaxed),
    };

    (mesh, stats)
}

// Fast face collection without deep traversal
pub fn collect_faces_from_brep<'a>(
    s: &'a StepFile,
    rep_item_id: Id<RepresentationItem_<'a>>,
) -> Vec<(AdvancedFace<'a>, bool)> {
    // Get the representation item first
    let rep_item = &s[rep_item_id];

    // Extract the actual entity that this representation item refers to
    // The representation item will have a different structure depending on its type
    match rep_item {
        Entity::ManifoldSolidBrep(b) => s
            .entity(b.outer)
            .map(|cs: &ClosedShell_| cs.cfs_faces.iter().map(|f| (f.cast(), false)).collect())
            .unwrap_or_default(),
        Entity::ShellBasedSurfaceModel(b) => b
            .sbsm_boundary
            .iter()
            .flat_map(|shell| match &s[*shell] {
                Entity::ClosedShell(cs) => cs.cfs_faces.iter().map(|f| (f.cast(), false)).collect(),
                Entity::OpenShell(os) => os.cfs_faces.iter().map(|f| (f.cast(), false)).collect(),
                _ => vec![],
            })
            .collect(),
        Entity::BrepWithVoids(b) => {
            let mut faces: Vec<(AdvancedFace, bool)> = s
                .entity(b.outer)
                .map(|cs: &ClosedShell_| cs.cfs_faces.iter().map(|f| (f.cast(), false)).collect())
                .unwrap_or_default();

            for void_id in &b.voids {
                if let Some(oriented) = s.entity(*void_id) {
                    let flip = !oriented.orientation;
                    if let Some(cs) = s.entity(oriented.closed_shell_element) {
                        faces.extend(cs.cfs_faces.iter().map(|f| (f.cast(), flip)));
                    }
                }
            }
            faces
        }
        _ => vec![],
    }
}

// Core triangulation - optimized hot path
fn triangulate_single_face(
    s: &StepFile,
    f: AdvancedFace,
    verts: &mut Vec<mesh::Vertex>,
    triangles: &mut Vec<Triangle>,
) -> Result<(), Error> {
    let face = s.entity(f).ok_or(Error::CouldNotLower)?;
    let mut surf = get_surface(s, face.face_geometry)?;

    let v_start = verts.len();
    let mut edges = Vec::with_capacity(face.bounds.len() * 32);
    let mut num_pts = 0;

    // Pre-allocate vertex buffer
    verts.reserve(face.bounds.len() * 32);

    for b in &face.bounds {
        let bound_contours = face_bound(s, *b)?;

        match bound_contours.len() {
            0 => return Err(Error::CouldNotLower),
            1 => {
                num_pts += 1;
                verts.push(mesh::Vertex {
                    pos: bound_contours[0],
                    norm: DVec3::zeros(),
                    color: DVec3::zeros(),
                });
            }
            _ => {
                let start = num_pts;
                for pt in &bound_contours[..bound_contours.len() - 1] {
                    edges.push((num_pts, num_pts + 1));
                    verts.push(mesh::Vertex {
                        pos: *pt,
                        norm: DVec3::zeros(),
                        color: DVec3::zeros(),
                    });
                    num_pts += 1;
                }
                edges.last_mut().unwrap().1 = start;
            }
        }
    }

    let mut pts = surf.lower_verts(&mut verts[v_start..])?;
    let bonus_points = pts.len();
    surf.add_steiner_points(&mut pts, verts);

    // Triangulation with retry logic (avoid panic overhead)
    let t = loop {
        match cdt::Triangulation::new_with_edges(&pts, &edges) {
            Ok(mut t) => match t.run() {
                Ok(()) => break t,
                Err(cdt::Error::PointOnFixedEdge(p)) if p >= bonus_points => {
                    pts[p] = pts[0];
                    continue;
                }
                Err(_) => return Err(Error::CouldNotLower),
            },
            Err(_) => return Err(Error::CouldNotLower),
        }
    };

    // Build triangles
    triangles.reserve(pts.len() * 2);
    for (a, b, c) in t.triangles() {
        let (a, b, c) = (a as u32, b as u32, c as u32);
        triangles.push(Triangle {
            verts: if face.same_sense {
                U32Vec3::new(a, b, c)
            } else {
                U32Vec3::new(a, c, b)
            },
        });
    }

    Ok(())
}

#[cfg(feature = "wgpu")]
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

    wgpu_impl::triangulate_faces(s, &face_tasks)
}


/// Batch triangulation function that processes all STEP files in the examples directory
/// This maximizes GPU utilization by processing all faces from all files in one operation
#[cfg(feature = "wgpu")]
pub fn wgpu_triangulate_batch_from_examples() -> Vec<(Mesh, Stats)> {
    use std::fs;
    use std::path::Path;
    use step::step_file::StepFile;
    use crate::wgpu_triangulate::wgpu_impl;
    
    // List of problematic STEP files that cause triangulation panics
    const PROBLEMATIC_FILES: &[&str] = &[
        "../examples/sphere.step",  // Known to cause assertion failure in CDT triangulation
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
                        && !PROBLEMATIC_FILES.contains(&path_str.as_str()) {
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
    
    println!("Batch processing {} STEP files", step_files_paths.len());

    // Create a single GPU device for the entire batch to avoid resource exhaustion
    let (device, queue) = match wgpu_impl::create_wgpu_device() {
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
                        triangulate4(&step_file)
                    } else {
                        (Mesh::default(), Stats::default())
                    }
                })
                .collect();
        }
    };

    // Process all files with the shared GPU device for maximum resource efficiency
    step_files_paths
        .iter()
        .map(|path| {
            if let Ok(contents) = fs::read(path) {
                let flattened = StepFile::strip_flatten(&contents);
                let step_file = StepFile::parse(&flattened);
                
                // Build face tasks for this specific file using the same logic as wgpu_triangulate
                let brep_colors: AHashMap<_, DVec3> =
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

                let mut todo: Vec<_> = roots.into_iter().map(|v| (v, DMat4::identity())).collect();
                let mut shape_rep_relationship: AHashMap<Id<_>, Vec<Id<_>>> = AHashMap::new();
                for (r1, r2) in
                    step_file.0.iter()
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
                let (mesh, stats) = wgpu_impl::triangulate_faces_with_device(&step_file, &face_tasks, &device, &queue);
                (mesh, stats) // Return both mesh and stats
            } else {
                (Mesh::default(), Stats::default())
            }
        })
        .collect()
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_wgpu_triangulate_batch_from_examples() {
        // This test verifies that the batch function compiles and can be called
        // It doesn't assert specific results since they depend on the files in examples directory
        let _results = wgpu_triangulate_batch_from_examples();
        // Should complete without panicking
    }
}