pub mod curve;
pub mod mesh;
pub mod stats;
pub mod surface;
pub mod triangulate;

#[cfg(feature = "wgpu")]
pub mod wgpu_triangulate;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Could not lower point to 2D for triangulation")]
    CouldNotLower,

    #[error("Point is located on a fixed edge but is not its endpoint")]
    PointOnFixedEdge,

    #[error("There are no more points left to triangulate")]
    NoMorePoints,

    #[error("Fixed edges cross each other")]
    CrossingFixedEdge,

    #[error("Input cannot be empty")]
    EmptyInput,

    #[error("Input cannot contain NaN or infinity")]
    InvalidInput,

    #[error("Edge must index into point array and have different src and dst")]
    InvalidEdge,

    #[error("Contours must be closed")]
    OpenContour,

    #[error("Too few points")]
    TooFewPoints,

    #[error("Could not find initial seed")]
    CannotInitialize,

    #[error("Escaped wedge when searching fixed edge")]
    WedgeEscape,

    #[error("Could not convert into a Surface")]
    UnknownSurfaceType,

    #[error("Could not convert into a Curve")]
    UnknownCurveType,

    #[error("Closed NURBS and b-spline surfaces are not implemented")]
    ClosedSurface,

    #[error("Self-intersecting NURBS and b-spline surfaces are not implemented")]
    SelfIntersectingSurface,

    #[error("Closed NURBS and b-spline curves are not implemented")]
    ClosedCurve,
    #[error("hull edge destination mismatch")]
    HullMismatch,

    #[error("Self-intersecting NURBS and b-spline curves are not implemented")]
    SelfIntersectingCurve,
}
