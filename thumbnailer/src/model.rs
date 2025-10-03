use nalgebra_glm as glm;
use triangulate::mesh::Mesh;

pub struct Model {
    pub mesh: Mesh,
}

impl Model {
    pub fn new(mesh: Mesh) -> Self {
        Self { mesh }
    }

    pub fn scale_model_to_unit_cube(&mut self) {
        // Calculate the bounding box of the mesh
        let mut min_corner = glm::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_corner = glm::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for vertex in &self.mesh.verts {
            let position = glm::Vec3::new(
                vertex.pos.x as f32,
                vertex.pos.y as f32,
                vertex.pos.z as f32,
            );
            min_corner = glm::vec3(
                min_corner.x.min(position.x),
                min_corner.y.min(position.y),
                min_corner.z.min(position.z),
            );
            max_corner = glm::vec3(
                max_corner.x.max(position.x),
                max_corner.y.max(position.y),
                max_corner.z.max(position.z),
            );
        }

        // Calculate the center and size
        let center_point = (min_corner + max_corner) * 0.5;
        let size_vector = max_corner - min_corner;
        let max_dimension = size_vector.x.max(size_vector.y).max(size_vector.z);

        if max_dimension > 0.0 {
            let scale_factor = 1.0 / max_dimension;
            let scale_matrix = glm::scaling(&glm::vec3(scale_factor, scale_factor, scale_factor));
            let translation_matrix = glm::translation(&-center_point);
            let transform_matrix = scale_matrix * translation_matrix;

            // Apply transformation to all vertices and normals
            for vertex in &mut self.mesh.verts {
                // Transform position
                let pos_vec4 = glm::vec4(
                    vertex.pos.x as f32,
                    vertex.pos.y as f32,
                    vertex.pos.z as f32,
                    1.0,
                );
                let transformed_pos = transform_matrix * pos_vec4;
                vertex.pos.x = transformed_pos.x as f64;
                vertex.pos.y = transformed_pos.y as f64;
                vertex.pos.z = transformed_pos.z as f64;

                // Transform normal
                let norm_vec3 = glm::vec3(
                    vertex.norm.x as f32,
                    vertex.norm.y as f32,
                    vertex.norm.z as f32,
                );
                // For uniform scaling, we can just re-normalise after transformation
                let transformed_norm = glm::normalize(&norm_vec3);
                vertex.norm.x = transformed_norm.x as f64;
                vertex.norm.y = transformed_norm.y as f64;
                vertex.norm.z = transformed_norm.z as f64;
            }
        }
    }
}
