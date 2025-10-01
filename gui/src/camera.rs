use glm::{Mat4, Vec2, Vec3, Vec4};
use nalgebra_glm as glm;
use winit::event::MouseButton;

use triangulate::mesh::Vertex;

#[derive(Copy, Clone, Debug)]
enum MouseState {
    Unknown,
    Free(Vec2),
    Rotate(Vec2),
    Pan(Vec2, Vec3),
}

pub struct Camera {
    /// Aspect ratio of the window
    width: f32,
    height: f32,

    /// Pitch as an Euler angle
    pitch: f32,

    /// Yaw as an Euler angle
    yaw: f32,

    /// Model scale
    scale: f32,

    /// Center of view volume
    center: Vec3,

    mouse: MouseState,
}

impl Camera {
    pub fn new(width: f32, height: f32) -> Self {
        Camera {
            width,
            height,
            // Isometric projection: 45° around Y axis, ~35.26° around X axis
            // pitch controls rotation around Y axis: glm::rotate_y(&i, self.pitch)
            // yaw controls rotation around X axis: glm::rotate_x(&i, self.yaw)
            pitch: (45.0_f32).to_radians(),   // 45 degrees around Y axis
            yaw: (35.26_f32).to_radians(),   // ~35.26 degrees around X axis  
            scale: 1.0,
            center: Vec3::zeros(),
            mouse: MouseState::Unknown,
        }
    }

    pub fn mouse_pressed(&mut self, button: MouseButton) {
        // If we were previously free, then switch to panning or rotating
        if let MouseState::Free(pos) = &self.mouse {
            match button {
                MouseButton::Left => Some(MouseState::Rotate(*pos)),
                MouseButton::Right => Some(MouseState::Pan(*pos, self.mouse_pos(*pos))),
                _ => None,
            }
            .map(|m| self.mouse = m);
        }
    }
    pub fn mouse_released(&mut self, button: MouseButton) {
        match &self.mouse {
            MouseState::Rotate(pos) if button == MouseButton::Left => Some(MouseState::Free(*pos)),
            MouseState::Pan(pos, ..) if button == MouseButton::Right => {
                Some(MouseState::Free(*pos))
            }
            _ => None,
        }
        .map(|m| self.mouse = m);
    }

    pub fn mat(&self) -> Mat4 {
        self.view_matrix() * self.model_matrix()
    }

    pub fn mat_i(&self) -> Mat4 {
        (self.view_matrix() * self.model_matrix())
            .try_inverse()
            .expect("Failed to invert mouse matrix")
    }

    /// Converts a normalized mouse position into 3D
    pub fn mouse_pos(&self, pos_norm: Vec2) -> Vec3 {
        (self.mat_i() * Vec4::new(pos_norm.x, pos_norm.y, 0.0, 1.0)).xyz()
    }

    pub fn mouse_move(&mut self, new_pos: Vec2) {
        let x_norm = 2.0 * (new_pos.x / self.width - 0.5);
        let y_norm = -2.0 * (new_pos.y / self.height - 0.5);
        let new_pos = Vec2::new(x_norm, y_norm);

        // Pan or rotate depending on current mouse state
        match &self.mouse {
            MouseState::Pan(_pos, orig) => {
                let current_pos = self.mouse_pos(new_pos);
                let delta_pos = orig - current_pos;
                self.center += delta_pos;
            }
            MouseState::Rotate(pos) => {
                let delta = new_pos - *pos;
                self.spin(delta.x * 3.0, -delta.y * 3.0 * self.height / self.width);
            }
            _ => (),
        }

        // Store new mouse position
        match &mut self.mouse {
            MouseState::Free(pos) | MouseState::Pan(pos, ..) | MouseState::Rotate(pos) => {
                *pos = new_pos
            }
            MouseState::Unknown => self.mouse = MouseState::Free(new_pos),
        }
    }

    pub fn mouse_scroll(&mut self, delta: f32) {
        if let MouseState::Free(pos) = self.mouse {
            self.scale(1.0 + delta / 200.0, pos);
        }
    }

    pub fn fit_verts(&mut self, verts: &[Vertex]) {
        if verts.is_empty() {
            return;
        }
        
        // Calculate bounding box - vertex positions are f64, cast to f32
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        
        for vertex in verts {
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
        
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let dz = max_z - min_z;
        
        // Calculate model center
        self.center = Vec3::new(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        );
        
        // Find the maximum dimension to ensure the longest dimension fills as much of the frame as possible
        let max_dimension = dx.max(dy).max(dz);
        
        // For the GUI's scaling approach, we want to make the max dimension fill ~98% of view
        // This means the scale should be such that max_dimension * scale = 0.98 (98% of normalized view)
        self.scale = 0.98 / max_dimension;  // Direct calculation for 98% fill - more aggressive
    }

    pub fn set_size(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    pub fn model_matrix(&self) -> Mat4 {
        let i = Mat4::identity();
        // The transforms below are applied bottom-to-top when thinking about
        // the model, i.e. it's translated, then scaled, then rotated, etc.

        // Scale to compensate for model size
        glm::scale(&i, &Vec3::new(self.scale, self.scale, self.scale)) *

        // Rotation!
        glm::rotate_x(&i, self.yaw) *
        glm::rotate_y(&i, self.pitch) *

        // Recenter model
        glm::translate(&i, &-self.center)
    }

    /// Returns a matrix which compensates for window aspect ratio and clipping
    pub fn view_matrix(&self) -> Mat4 {
        let i = Mat4::identity();
        // The Z clipping range is 0-1, so push forward
        glm::translate(&i, &Vec3::new(0.0, 0.0, 0.5)) *

        // Scale to compensate for aspect ratio and reduce Z scale to improve
        // clipping
        glm::scale(&i, &Vec3::new(1.0, self.width / self.height, 0.1))
    }

    pub fn spin(&mut self, dx: f32, dy: f32) {
        self.pitch += dx;
        self.yaw += dy;
    }

    pub fn scale(&mut self, value: f32, pos: Vec2) {
        let start_pos = self.mouse_pos(pos);
        self.scale *= value;
        let end_pos = self.mouse_pos(pos);

        let delta = start_pos - end_pos;
        let mut delta_mouse = (self.mat() * delta.to_homogeneous()).xyz();
        delta_mouse.z = 0.0;

        self.center += (self.mat_i() * delta_mouse.to_homogeneous()).xyz();
    }
}
