//! Static geometry previewer (feature `preview`). Opens a window via
//! `minifb`, blits a per-pixel CSG sample of the geometry into it,
//! waits for the user to close it (Esc or window-X).
//!
//! Per-pixel CSG sampler: for every output pixel, evaluate the
//! corresponding world coordinate, ask
//! [`crate::geometry::ray::find_cell`] which cell contains it, look
//! up the cell's material, and write the material's RGB colour to
//! the framebuffer. Top-down slice at a chosen `z` is the canonical
//! reactor / shielding cross-section view.
//!
//! The previewer reuses the production CSG path — no separate
//! geometry description needed. If the simulation runs, the preview
//! reflects exactly the geometry it runs on.

pub mod render;

pub use render::{
    LegendEntry, MaterialPalette, NamedMaterial, Viewport, legend_from_materials,
    preview_geometry, render_top_down, show_window,
};
