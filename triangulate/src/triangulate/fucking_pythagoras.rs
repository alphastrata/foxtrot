use super::*;
use std::collections::HashMap;
use std::convert::TryInto;

use glm::{DMat4, DVec3, DVec4, U32Vec3};
use nalgebra_glm as glm;

use crate::{
    Error,
    curve::Curve,
    mesh,
    mesh::{Mesh, Triangle},
    stats::Stats,
    surface::Surface,
};
use nurbs::{BSplineSurface, KnotVector, NURBSSurface, SampledSurface};
use step::{
    ap214,
    ap214::Entity,
    id::Id,
    step_file::{FromEntity, StepFile},
};

struct EntityCache<'a> {
    step_file: &'a StepFile<'a>,
    // Cache commonly accessed entity types
    cartesian_points: HashMap<Id<CartesianPoint_<'a>>, DVec3>,
    directions: HashMap<Id<Direction_<'a>>, DVec3>,
    axis2_placements: HashMap<Id<Axis2Placement3d_<'a>>, (DVec3, DVec3, DVec3)>,
    surfaces: HashMap<Id<Surface_<'a>>, Surface>,
    oriented_edges: HashMap<Id<OrientedEdge_<'a>>, &'a OrientedEdge_<'a>>,
    edge_curves: HashMap<Id<EdgeCurve_<'a>>, &'a EdgeCurve_<'a>>,
    vertex_points: HashMap<Id<VertexPoint_<'a>>, DVec3>,
}

impl<'a> EntityCache<'a> {
    fn new(s: &'a StepFile) -> Self {
        // Pre-populate common lookups
        let mut cache = EntityCache {
            step_file: s,
            cartesian_points: HashMap::new(),
            directions: HashMap::new(),
            axis2_placements: HashMap::new(),
            surfaces: HashMap::new(),
            oriented_edges: HashMap::new(),
            edge_curves: HashMap::new(),
            vertex_points: HashMap::new(),
        };

        // Build lookup tables by scanning once
        for (idx, entity) in s.0.iter().enumerate() {
            match entity {
                Entity::CartesianPoint(p) => {
                    let id = Id::new(idx);
                    let point =
                        DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0);
                    cache.cartesian_points.insert(id, point);
                }
                Entity::Direction(d) => {
                    let id = Id::new(idx);
                    let dir = DVec3::new(
                        d.direction_ratios[0],
                        d.direction_ratios[1],
                        d.direction_ratios[2],
                    );
                    cache.directions.insert(id, dir);
                }
                Entity::OrientedEdge(oe) => {
                    cache.oriented_edges.insert(Id::new(idx), oe);
                }
                Entity::EdgeCurve(ec) => {
                    cache.edge_curves.insert(Id::new(idx), ec);
                }
                Entity::VertexPoint(vp) => {
                    let id = Id::new(idx);
                    // Get cartesian point directly from step file
                    let cp = cartesian_point(cache.step_file, vp.vertex_geometry.cast());
                    cache.vertex_points.insert(id, cp);
                }
                _ => {}
            }
        }

        cache
    }

    fn cartesian_point(&self, id: Id<CartesianPoint_>) -> DVec3 {
        *self
            .cartesian_points
            .get(&id)
            .expect("CartesianPoint not in cache")
    }

    fn direction(&self, id: Direction) -> DVec3 {
        *self.directions.get(&id).expect("Direction not in cache")
    }

    fn vertex_point(&self, v: Vertex) -> DVec3 {
        match self.vertex_points.get(&v.cast()) {
            Some(&point) => point,
            None => {
                // Fallback: get the vertex point directly from the step file
                let vertex_point_entity: &VertexPoint_<'a> = self
                    .step_file
                    .entity(v.cast())
                    .expect("VertexPoint not in file");
                let cartesian_point_id: Id<CartesianPoint_<'a>> =
                    vertex_point_entity.vertex_geometry.cast();
                let cp_entity: &CartesianPoint_<'a> = self
                    .step_file
                    .entity(cartesian_point_id)
                    .expect("Cartesian point not in file");
                DVec3::new(
                    cp_entity.coordinates[0].0,
                    cp_entity.coordinates[1].0,
                    cp_entity.coordinates[2].0,
                )
            }
        }
    }

    fn axis2_placement_3d(&mut self, id: Id<Axis2Placement3d_<'a>>) -> (DVec3, DVec3, DVec3) {
        if let Some(cached) = self.axis2_placements.get(&id) {
            return *cached;
        }

        let a = self
            .step_file
            .entity(id)
            .expect("Could not get Axis2Placement3d");
        let location = self.cartesian_point(a.location);
        let axis = self.direction(a.axis.expect("Missing axis"));
        let ref_direction = match a.ref_direction {
            None => DVec3::new(1.0, 0.0, 0.0),
            Some(r) => self.direction(r),
        };

        let result = (location, axis, ref_direction);
        self.axis2_placements.insert(id, result);
        result
    }

    fn get_surface(&mut self, surf_id: Id<Surface_<'a>>) -> Result<Surface, Error> {
        // Check cache first
        if let Some(cached) = self.surfaces.get(&surf_id) {
            return Ok(cached.clone());
        }

        let surf = match &self.step_file[surf_id] {
            Entity::CylindricalSurface(c) => {
                let (location, axis, ref_direction) = self.axis2_placement_3d(c.position);
                Surface::new_cylinder(axis, ref_direction, location, c.radius.0.0.0)
            }
            Entity::ToroidalSurface(c) => {
                let (location, axis, _) = self.axis2_placement_3d(c.position);
                Surface::new_torus(location, axis, c.major_radius.0.0.0, c.minor_radius.0.0.0)
            }
            Entity::Plane(p) => {
                let (location, axis, ref_direction) = self.axis2_placement_3d(p.position);
                Surface::new_plane(axis, ref_direction, location)
            }
            Entity::ConicalSurface(c) => {
                let (location, axis, ref_direction) = self.axis2_placement_3d(c.position);
                Surface::new_cone(axis, ref_direction, location, c.semi_angle.0)
            }
            Entity::SphericalSurface(c) => {
                let (location, _, _) = self.axis2_placement_3d(c.position);
                Surface::new_sphere(location, c.radius.0.0.0)
            }
            Entity::BSplineSurfaceWithKnots(b) => {
                return self.build_bspline_surface(b);
            }
            Entity::ComplexEntity(v) if v.len() == 2 => {
                return self.build_nurbs_surface(v.as_slice());
            }
            _ => return Err(Error::UnknownSurfaceType),
        };

        self.surfaces.insert(surf_id, surf.clone());
        Ok(surf)
    }

    fn build_bspline_surface(&mut self, b: &BSplineSurfaceWithKnots_) -> Result<Surface, Error> {
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

        let control_points_list = self.control_points_2d(&b.control_points_list);
        let surf = BSplineSurface::new(
            !b.u_closed.0.unwrap(),
            !b.v_closed.0.unwrap(),
            u_knot_vec,
            v_knot_vec,
            control_points_list,
        );
        Ok(Surface::BSpline(SampledSurface::new(surf)))
    }

    fn build_nurbs_surface(&mut self, v: &[Entity]) -> Result<Surface, Error> {
        let bspline = if let Entity::BSplineSurfaceWithKnots(b) = &v[0] {
            b
        } else {
            return Err(Error::UnknownCurveType);
        };
        let rational = if let Entity::RationalBSplineSurface(b) = &v[1] {
            b
        } else {
            return Err(Error::UnknownCurveType);
        };

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

        let control_points_list = self
            .control_points_2d(&bspline.control_points_list)
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

    fn control_points_1d(&self, row: &Vec<CartesianPoint>) -> Vec<DVec3> {
        row.iter().map(|p| self.cartesian_point(*p)).collect()
    }

    fn control_points_2d(&self, rows: &Vec<Vec<CartesianPoint>>) -> Vec<Vec<DVec3>> {
        rows.iter().map(|row| self.control_points_1d(row)).collect()
    }

    // Create a copy suitable for thread-local use
    fn clone_for_thread(&self) -> EntityCache<'a> {
        // Use the existing data but create a fresh cache for surfaces since they're computed per face
        EntityCache {
            step_file: self.step_file,
            cartesian_points: self.cartesian_points.clone(),
            directions: self.directions.clone(),
            axis2_placements: HashMap::new(), // Start with empty cache - will be populated as needed
            surfaces: HashMap::new(), // Start with empty cache - will be populated as needed
            oriented_edges: self.oriented_edges.clone(),
            edge_curves: self.edge_curves.clone(),
            vertex_points: self.vertex_points.clone(),
        }
    }
}

// Optimized edge loop that uses cache
fn edge_loop2<'a>(
    s: &'a StepFile,
    cache: &EntityCache<'a>,
    edge_list: &[OrientedEdge<'a>],
) -> Result<Vec<DVec3>, Error> {
    let mut out = Vec::with_capacity(edge_list.len() * 8);

    for (i, &edge_id) in edge_list.iter().enumerate() {
        if i > 0 {
            out.pop();
        }

        let edge = cache
            .oriented_edges
            .get(&edge_id)
            .expect("OrientedEdge not in cache");
        let edge_curve = cache
            .edge_curves
            .get(&edge.edge_element.cast())
            .expect("EdgeCurve not in cache");

        // Get curve points
        let (start, end) = if edge.orientation {
            (edge_curve.edge_start, edge_curve.edge_end)
        } else {
            (edge_curve.edge_end, edge_curve.edge_start)
        };

        let u = cache.vertex_point(start);
        let v = cache.vertex_point(end);

        // Build curve (this still needs optimization but is less critical)
        let curve = curve2(
            s,
            cache,
            edge_curve,
            edge_curve.edge_geometry,
            edge.orientation,
        )?;
        out.extend(curve.build(u, v));
    }

    Ok(out)
}

// Simplified curve builder (focused on common cases)
fn curve2<'a>(
    s: &'a StepFile,
    cache: &EntityCache<'a>,
    edge_curve: &EdgeCurve_,
    curve_id: ap214::Curve,
    orientation: bool,
) -> Result<Curve, Error> {
    Ok(match &s[curve_id] {
        Entity::Line(_) => Curve::new_line(),
        Entity::Circle(c) => {
            // For now, use the original axis2_placement_3d function since cache is immutable here
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
        Entity::SurfaceCurve(v) => curve2(s, cache, edge_curve, v.curve_3d, orientation)?,
        Entity::SeamCurve(v) => curve2(s, cache, edge_curve, v.curve_3d, orientation)?,
        // Fall back to original for complex curves
        _ => curve(s, edge_curve, curve_id, orientation)?,
    })
}

fn face_bound2<'a>(
    s: &'a StepFile,
    cache: &EntityCache<'a>,
    b: FaceBound<'a>,
) -> Result<Vec<DVec3>, Error> {
    let (bound, orientation) = match &s[b] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        e => panic!("Could not get bound from {:?}", e),
    };

    match &s[bound] {
        Entity::EdgeLoop(e) => {
            let mut d = edge_loop2(s, cache, &e.edge_list)?;
            if !orientation {
                d.reverse()
            }
            Ok(d)
        }
        Entity::VertexLoop(v) => Ok(vec![cache.vertex_point(v.loop_vertex)]),
        e => panic!("{:?} is not an EdgeLoop", e),
    }
}

#[cfg(feature = "rayon")]
pub fn triangulate5(s: &StepFile) -> (Mesh, Stats) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ... (same setup as triangulate4 for colors, transforms, etc)
    let brep_colors: HashMap<_, DVec3> =
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
    let mut shape_rep_relationship: HashMap<Id<_>, Vec<Id<_>>> = HashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: HashMap<Id<_>, Vec<DMat4>> = HashMap::new();
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

    let num_shells = to_mesh.len();
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

    let total_faces = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let total_panics = AtomicUsize::new(0);

    let mesh = face_tasks
        .par_iter()
        .filter_map(|task| {
            let mut verts = Vec::with_capacity(128);
            let mut triangles = Vec::with_capacity(128);
            total_faces.fetch_add(1, Ordering::Relaxed);

            // Create a new cache for each task to avoid synchronization issues
            let mut cache = EntityCache::new(s);
            match triangulate_face_cached(s, &mut cache, task.face_id, &mut verts, &mut triangles) {
                Ok(()) => {
                    if task.flip_normal {
                        for v in &mut verts {
                            v.norm = -v.norm;
                        }
                    }

                    let mut mesh = Mesh {
                        verts: Vec::with_capacity(verts.len() * task.transforms.len()),
                        triangles: Vec::with_capacity(triangles.len() * task.transforms.len()),
                    };

                    for mat in &task.transforms {
                        let v_offset = mesh.verts.len() as u32;
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
                Err(_) => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
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

fn triangulate_face_cached<'a>(
    s: &'a StepFile<'a>,
    cache: &mut EntityCache<'a>,
    f: AdvancedFace<'a>,
    verts: &mut Vec<mesh::Vertex>,
    triangles: &mut Vec<Triangle>,
) -> Result<(), Error> {
    let face = s.entity(f).ok_or(Error::CouldNotLower)?;
    let mut surf = cache.get_surface(face.face_geometry)?;

    let v_start = verts.len();
    let mut edges = Vec::with_capacity(32);
    let mut num_pts = 0;
    verts.reserve(32);

    for b in &face.bounds {
        let bound_contours = face_bound2(s, cache, *b)?;
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

    triangles.reserve(pts.len() * 2);
    for (a, b, c) in t.triangles() {
        triangles.push(Triangle {
            verts: if face.same_sense {
                U32Vec3::new(a as u32, b as u32, c as u32)
            } else {
                U32Vec3::new(a as u32, c as u32, b as u32)
            },
        });
    }

    Ok(())
}

#[cfg(feature = "rayon")]
pub fn triangulate6(s: &StepFile) -> (Mesh, Stats) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Phase 1: Build lookup cache for the whole StepFile (like in triangulate5)
    let entity_cache = EntityCache::new(s);

    // ... (same setup as triangulate4 for colors, transforms, etc)
    let brep_colors: HashMap<_, DVec3> =
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
    let mut shape_rep_relationship: HashMap<Id<_>, Vec<Id<_>>> = HashMap::new();
    for (r1, r2) in
        s.0.iter()
            .filter_map(ShapeRepresentationRelationship_::try_from_entity)
            .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: HashMap<Id<_>, Vec<DMat4>> = HashMap::new();
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

    struct FaceTask<'a> {
        face_id: AdvancedFace<'a>,
        transforms: Vec<DMat4>,
        color: DVec3,
        flip_normal: bool,
    }

    let num_shells = to_mesh.len();
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

    // Phase 3: Parallel triangulation using pre-built cache
    let total_faces = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let total_panics = AtomicUsize::new(0);

    let mesh = face_tasks
        .par_iter()
        .filter_map(|task| {
            let mut verts = Vec::with_capacity(128);
            let mut triangles = Vec::with_capacity(128);
            total_faces.fetch_add(1, Ordering::Relaxed);

            // Use a clone of the pre-built cache for each task
            // This avoids rebuilding the cache but each task has its own copy
            let mut local_cache = entity_cache.clone_for_thread();
            match triangulate_face_cached(
                s,
                &mut local_cache,
                task.face_id,
                &mut verts,
                &mut triangles,
            ) {
                Ok(()) => {
                    if task.flip_normal {
                        for v in &mut verts {
                            v.norm = -v.norm;
                        }
                    }

                    let mut mesh = Mesh {
                        verts: Vec::with_capacity(verts.len() * task.transforms.len()),
                        triangles: Vec::with_capacity(triangles.len() * task.transforms.len()),
                    };

                    for mat in &task.transforms {
                        let v_offset = mesh.verts.len() as u32;
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
                Err(_) => {
                    total_errors.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .reduce(|| Mesh::default(), |a, b| Mesh::combine(a, b));

    let stats = Stats {
        num_shells,
        num_faces: total_faces.load(Ordering::Relaxed),
        num_errors: total_errors.load(Ordering::Relaxed),
        num_panics: total_panics.load(Ordering::Relaxed),
    };

    (mesh, stats)
}
