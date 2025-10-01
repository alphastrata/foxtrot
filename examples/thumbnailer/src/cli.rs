use clap::Parser;

#[derive(Parser, Clone, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Input STEP file to render
    #[clap(short = 'i', long = "input", value_parser, required = true)]
    pub input: String,

    /// Output PNG file path
    #[clap(short = 'o', long = "output", value_parser, required = true)]
    pub output: String,

    /// Thumbnail size in pixels (width and height)
    #[clap(short = 's', long = "thumbnail-size", default_value = "512")]
    pub thumbnail_size: u32,

    /// Whether to render with transparent background
    #[clap(long)]
    pub transparent: bool,

    /// Whether to auto-crop the output
    #[clap(long)]
    pub crop: bool,

    /// Render mode - single isometric (iso) or 4-view composite (composite)
    #[clap(long, default_value = "iso", value_enum)]
    pub mode: RenderMode,

    /// Sets the camera view for the render
    #[clap(long, value_enum, default_value = "isometric")]
    pub view: CameraView,

    /// Triangulation engine to use (requires OCCT feature)
    #[cfg(feature = "occt")]
    #[clap(long, value_enum, default_value = "foxtrot")]
    pub engine: TriangulationEngine,

    /// Triangulation engine to use
    #[cfg(not(feature = "occt"))]
    #[clap(long, value_enum, default_value = "foxtrot")]
    pub engine: TriangulationEngine,

    /// Target reduction ratio for mesh decimation (0.0 - 1.0, where 0.5 = 50% reduction)
    #[clap(long = "decimation-ratio", default_value = "0.25")]
    pub decimation_ratio: f32,

    /// Target error tolerance for mesh decimation (higher = more aggressive simplification)
    #[clap(long = "decimation-error", default_value = "0.02")]
    pub decimation_error: f32,

    /// Linear deflection for OCCT triangulation (smaller = more triangles)
    #[cfg(feature = "occt")]
    #[clap(long = "occt-linear-deflection", default_value = "0.01")]
    pub occt_linear_deflection: f64,

    /// Angular deflection for OCCT triangulation (smaller = more accurate)
    #[cfg(feature = "occt")]
    #[clap(long = "occt-angular-deflection", default_value = "0.5")]
    pub occt_angular_deflection: f64,

    /// Whether to use shaded rendering instead of wireframe
    #[clap(long, default_value_t = false)]
    pub shaded: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum RenderMode {
    Iso,
    Composite,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CameraView {
    Isometric,
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum TriangulationEngine {
    Foxtrot,
    #[cfg(feature = "occt")]
    Occt,
}
