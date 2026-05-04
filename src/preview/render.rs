use crate::geometry::{Cell, Surface, Vec3, ray};

/// World-space window for a top-down slice render.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    /// `z` coordinate at which the slice is taken. Geometry built in
    /// 2D (no `PlaneZ` constraints) is invariant under this.
    pub z_slice: f64,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// Square viewport centred at the origin with `half_extent` cm
    /// in both x and y at the chosen `z`. Convenience for
    /// reactor-cross-section previews.
    pub fn square_centered(half_extent: f64, z_slice: f64, side_px: u32) -> Self {
        Self {
            x_min: -half_extent,
            x_max: half_extent,
            y_min: -half_extent,
            y_max: half_extent,
            z_slice,
            width: side_px,
            height: side_px,
        }
    }
}

/// Per-material colour palette. Indexed by material index returned
/// from the user's `cell_to_material` closure. Falls back to `void`
/// for indices past the end of the palette and for pixels that don't
/// map to any cell at all.
#[derive(Debug, Clone)]
pub struct MaterialPalette {
    pub colors: Vec<[u8; 3]>,
    pub void: [u8; 3],
}

impl Default for MaterialPalette {
    fn default() -> Self {
        Self {
            colors: vec![
                [200, 80, 60],   // 0 — fuel (warm red)
                [120, 120, 120], // 1 — clad (grey)
                [60, 130, 220],  // 2 — water (cool blue)
                [255, 220, 60],  // 3 — control / boron (yellow)
                [60, 200, 100],  // 4 — moderator/reflector (green)
                [180, 100, 200], // 5 — instrumentation (purple)
                [220, 220, 220], // 6 — steel (light grey)
                [80, 60, 40],    // 7 — concrete (dark brown)
            ],
            void: [12, 12, 16],
        }
    }
}

/// Render a top-down CSG slice into a flat `u32` framebuffer in
/// `0x00RRGGBB` (minifb-native) format. Caller decides what to do
/// with it — pass to [`show_window`], save to PNG via the user's
/// preferred encoder, etc.
pub fn render_top_down(
    cells: &[Cell],
    surfaces: &[Surface],
    cell_to_material: impl Fn(usize) -> usize + Sync,
    palette: &MaterialPalette,
    viewport: &Viewport,
) -> Vec<u32> {
    let w = viewport.width as usize;
    let h = viewport.height as usize;
    let dx = (viewport.x_max - viewport.x_min) / viewport.width as f64;
    let dy = (viewport.y_max - viewport.y_min) / viewport.height as f64;
    let mut buf = vec![0u32; w * h];

    for py in 0..viewport.height {
        // Image y goes top-down; world y goes bottom-up.
        let world_y = viewport.y_max - (py as f64 + 0.5) * dy;
        for px in 0..viewport.width {
            let world_x = viewport.x_min + (px as f64 + 0.5) * dx;
            let pos = Vec3::new(world_x, world_y, viewport.z_slice);
            let color = match ray::find_cell(pos, surfaces, cells) {
                Some(idx) => {
                    let mat = cell_to_material(idx);
                    palette.colors.get(mat).copied().unwrap_or(palette.void)
                }
                None => palette.void,
            };
            buf[(py as usize) * w + (px as usize)] = pack_rgb(color);
        }
    }
    buf
}

#[inline]
fn pack_rgb([r, g, b]: [u8; 3]) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Open a minifb window, blit `buffer` into it, and block until the
/// user closes the window (X button) or presses Esc.
///
/// `buffer` must be exactly `width * height` `u32` values in
/// `0x00RRGGBB` format. Width and height are in pixels and must
/// match the viewport used to render the buffer.
pub fn show_window(buffer: &[u32], width: usize, height: usize, title: &str) {
    use minifb::{Key, Window, WindowOptions};

    let mut window = Window::new(title, width, height, WindowOptions::default())
        .unwrap_or_else(|e| panic!("failed to open preview window: {e}"));
    window.set_target_fps(30);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(buffer, width, height)
            .unwrap_or_else(|e| panic!("failed to blit framebuffer: {e}"));
    }
}
