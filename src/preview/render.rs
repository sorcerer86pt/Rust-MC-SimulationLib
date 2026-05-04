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

/// Open a minifb window with the given initial `viewport` and let
/// the user resize / zoom interactively. The `render` closure is
/// invoked once on open and again whenever the world-space view or
/// pixel resolution changes (window drag, scroll-wheel zoom, `R`
/// reset). Returns when the user closes the window or presses Esc.
///
/// Behaviour:
///   * Drag-resize the window — the cm-per-pixel ratio is held
///     constant, so a bigger window zooms *out* (more area visible)
///     and a smaller window zooms *in*. This matches the natural
///     "your screen is your viewport" intuition.
///   * Scroll wheel — multiplicative zoom around the viewport's
///     centre. Each notch shrinks (`up`) or expands (`down`) the
///     world-space extent by a constant factor.
///   * `R` — reset to the initial viewport.
///   * `Esc` or window-X — close.
pub fn show_window<F>(initial: Viewport, title: &str, mut render: F)
where
    F: FnMut(&Viewport) -> Vec<u32>,
{
    use minifb::{Key, Window, WindowOptions};

    let mut viewport = initial;
    let mut window = Window::new(
        title,
        viewport.width as usize,
        viewport.height as usize,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("failed to open preview window: {e}"));
    window.set_target_fps(30);

    let mut buffer = render(&viewport);
    let mut last_size = (viewport.width as usize, viewport.height as usize);
    let mut prev_r_pressed = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let cur_size = window.get_size();
        let mut needs_render = false;

        // Window drag-resize → constant cm/pixel zoom.
        if cur_size != last_size && cur_size.0 > 0 && cur_size.1 > 0 {
            let cx = (viewport.x_min + viewport.x_max) * 0.5;
            let cy = (viewport.y_min + viewport.y_max) * 0.5;
            // Use the initial cm-per-pixel as the anchor so the user
            // can drag back to the same scale they started at.
            let world_w = viewport.x_max - viewport.x_min;
            let world_h = viewport.y_max - viewport.y_min;
            let px_per_cm_x = viewport.width as f64 / world_w;
            let px_per_cm_y = viewport.height as f64 / world_h;
            let px_per_cm = px_per_cm_x.min(px_per_cm_y).max(1.0e-6);
            let new_world_w = cur_size.0 as f64 / px_per_cm;
            let new_world_h = cur_size.1 as f64 / px_per_cm;
            viewport.x_min = cx - new_world_w * 0.5;
            viewport.x_max = cx + new_world_w * 0.5;
            viewport.y_min = cy - new_world_h * 0.5;
            viewport.y_max = cy + new_world_h * 0.5;
            viewport.width = cur_size.0 as u32;
            viewport.height = cur_size.1 as u32;
            last_size = cur_size;
            needs_render = true;
        }

        // Scroll-wheel zoom around the viewport centre.
        if let Some((_, sy)) = window.get_scroll_wheel() {
            if sy.abs() > 0.0 {
                let factor = if sy > 0.0 { 0.85 } else { 1.0 / 0.85 };
                let cx = (viewport.x_min + viewport.x_max) * 0.5;
                let cy = (viewport.y_min + viewport.y_max) * 0.5;
                let hx = (viewport.x_max - viewport.x_min) * 0.5 * factor;
                let hy = (viewport.y_max - viewport.y_min) * 0.5 * factor;
                viewport.x_min = cx - hx;
                viewport.x_max = cx + hx;
                viewport.y_min = cy - hy;
                viewport.y_max = cy + hy;
                needs_render = true;
            }
        }

        // 'R' resets the view (debounced — fire on press, not hold).
        let r_now = window.is_key_down(Key::R);
        if r_now && !prev_r_pressed {
            viewport = initial;
            // Window size doesn't follow the viewport reset
            // (minifb has no programmatic resize), so respect the
            // current physical window size by treating it as a
            // resize event the next iteration.
            last_size = (0, 0);
            needs_render = true;
        }
        prev_r_pressed = r_now;

        if needs_render {
            buffer = render(&viewport);
        }

        window
            .update_with_buffer(&buffer, viewport.width as usize, viewport.height as usize)
            .unwrap_or_else(|e| panic!("failed to blit framebuffer: {e}"));
    }
}
