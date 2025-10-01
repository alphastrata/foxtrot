use crate::cli::CameraView;
use nalgebra_glm as glm;
use triangulate::mesh::Vertex;

pub struct Camera {
    pub projection_matrix: glm::Mat4,
    pub view_matrix: glm::Mat4,
    pub position: glm::Vec3,
}

impl Camera {
    pub fn new() -> Self {
        let initial_position = glm::vec3(0.0, 0.0, 5.0);
        let look_target = glm::vec3(0.0, 0.0, 0.0);
        let up_direction = glm::vec3(0.0, 1.0, 0.0);

        let view_matrix = glm::look_at(&initial_position, &look_target, &up_direction);

        // --- CHANGE: Use a default orthographic projection ---
        let projection_matrix = glm::ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);

        Self {
            projection_matrix,
            view_matrix,
            position: initial_position,
        }
    }

    // --- REMOVED: This function is no longer needed for orthographic projection ---
    // pub fn update_projection_matrix(&mut self, aspect_ratio: f32) { ... }

    pub fn set_isometric_view(&mut self) {
        // Set up an isometric view (45 degree angles)
        self.position = glm::vec3(5.0, 5.0, 5.0);
        let look_target = glm::vec3(0.0, 0.0, 0.0);
        let up_direction = glm::vec3(0.0, 1.0, 0.0); //QUESTION: What is step up?
        self.view_matrix = glm::look_at(&self.position, &look_target, &up_direction);
    }

    pub fn set_view_from_angles(&mut self, azimuth_degrees: f32, elevation_degrees: f32) {
        let radius = self.position.magnitude();
        let azimuth_radians = azimuth_degrees.to_radians();
        let elevation_radians = elevation_degrees.to_radians();

        let x = radius * azimuth_radians.cos() * elevation_radians.cos();
        let y = radius * elevation_radians.sin();
        let z = radius * azimuth_radians.sin() * elevation_radians.cos();

        self.position = glm::vec3(x, y, z);
        let look_target = glm::vec3(0.0, 0.0, 0.0);
        // FIXED: Use Y-up (0, 1, 0) instead of Z-up (0, 0, 1) to match the rest of the camera logic
        let up_direction = glm::vec3(0.0, 1.0, 0.0);
        self.view_matrix = glm::look_at(&self.position, &look_target, &up_direction);
    }

    pub fn fit_to_bounds(&mut self, vertices: &[Vertex], aspect_ratio: f32) {
        if vertices.is_empty() {
            return;
        }

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;

        for vertex in vertices.iter() {
            let x = vertex.pos.x as f32;
            let y = vertex.pos.y as f32;
            let z = vertex.pos.z as f32;

            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        let center = glm::vec3(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        );

        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        let size_z = max_z - min_z;
        let diagonal = (size_x * size_x + size_y * size_y + size_z * size_z).sqrt();
        let distance = if diagonal > 0.0 { diagonal * 1.5 } else { 5.0 };

        // --- CHANGE: Set up orthographic projection with extremely generous bounds ---
        // When viewing a 3D object from an isometric angle, the apparent size can increase
        // significantly due to the diagonal projection. We use a very generous zoom factor
        // to ensure the entire model is visible from any viewing angle.
        //
        // The diagonal of a unit cube is sqrt(3) ≈ 1.73, so we use a factor of 4.0
        // to ensure ample margin for all viewing angles.
        let max_dim = size_x.max(size_y).max(size_z);
        let half_dim = max_dim / 2.0;
        let zoom_factor = 1.25; // Add 50% padding around the model to accommodate diagonal viewing angles

        let left = -half_dim * zoom_factor * aspect_ratio;
        let right = half_dim * zoom_factor * aspect_ratio;
        let bottom = -half_dim * zoom_factor;
        let top = half_dim * zoom_factor;

        let near_plane = 0.0001;
        let far_plane = distance * 1.5;

        self.projection_matrix = glm::ortho(left, right, bottom, top, near_plane, far_plane);
        // --- End Change ---

        let azimuth = 45.0_f32.to_radians();
        let elevation = 35.0_f32.to_radians();

        let offset_x = distance * azimuth.cos() * elevation.cos();
        let offset_y = distance * elevation.sin();
        let offset_z = distance * azimuth.sin() * elevation.cos();

        self.position = glm::vec3(
            center.x + offset_x,
            center.y + offset_y,
            center.z + offset_z,
        );

        let look_target = center;
        let up_direction = glm::vec3(0.0, 1.0, 0.0);
        self.view_matrix = glm::look_at(&self.position, &look_target, &up_direction);

        // --- REMOVED: No longer need to call this separately ---
        // self.update_projection_matrix(aspect_ratio);
    }

    pub fn set_view(&mut self, view: &CameraView, center: &glm::Vec3) {
        let distance = (self.position - *center).magnitude();
        let up_y = glm::vec3(0.0, 1.0, 0.0); // Y-up vector
        let up_z = glm::vec3(0.0, 0.0, 1.0); // Z-up vector

        match view {
            CameraView::Isometric => {
                self.set_view_from_angles(45.0, 35.0);// set_view_from_angles handles its own matrix update
            }
            CameraView::Top => {
                self.position = *center + glm::vec3(0.0, distance, 0.0);
                self.view_matrix = glm::look_at(&self.position, center, &up_z);
            }
            CameraView::Bottom => {
                self.position = *center - glm::vec3(0.0, distance, 0.0);
                self.view_matrix = glm::look_at(&self.position, center, &up_z);
            }
            CameraView::Left => {
                self.position = *center - glm::vec3(distance, 0.0, 0.0);
                self.view_matrix = glm::look_at(&self.position, center, &up_y);
            }
            CameraView::Right => {
                self.position = *center + glm::vec3(distance, 0.0, 0.0);
                self.view_matrix = glm::look_at(&self.position, center, &up_y);
            }
            CameraView::Front => {
                self.position = *center + glm::vec3(0.0, 0.0, distance);
                self.view_matrix = glm::look_at(&self.position, center, &up_y);
            }
            CameraView::Back => {
                self.position = *center - glm::vec3(0.0, 0.0, distance);
                self.view_matrix = glm::look_at(&self.position, center, &up_y);
            }
        }
    }
}
