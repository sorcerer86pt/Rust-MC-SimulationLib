use crate::geometry::{Cell, Surface, Vec3, ray};

/// Anything that carries a human-readable material name. Implemented
/// out of the box for [`crate::transport::material::Material`] and
/// [`crate::photon::material::PhotonMaterial`] so callers can pass
/// their existing simulation material lists straight into the
/// previewer — no parallel "preview material" type to maintain.
pub trait NamedMaterial {
    fn name(&self) -> &str;
}

impl NamedMaterial for crate::transport::material::Material {
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(feature = "preview")]
impl NamedMaterial for crate::photon::material::PhotonMaterial {
    fn name(&self) -> &str {
        &self.name
    }
}

/// One row in the legend popup: a colour swatch + a text label.
#[derive(Debug, Clone)]
pub struct LegendEntry {
    pub label: String,
    pub color: [u8; 3],
}

impl LegendEntry {
    pub fn new(label: impl Into<String>, color: [u8; 3]) -> Self {
        Self {
            label: label.into(),
            color,
        }
    }
}

/// Heuristic colour for a material based on its name. Returns
/// `None` when no keyword matches; the caller is expected to fall
/// back to a default-palette colour.
///
/// Keyword groups (most-specific first — the first match wins):
///
///   * heavy water / D₂O / D2O                → darker blue
///   * light water / H₂O / H2O / "water"      → cool blue
///   * MOX / plutonium / PuO₂                 → bright orange
///   * UO₂ / uranium / "fuel" (+ burn-up tag) → fresh red / mid orange / burnt purple
///   * zircaloy / clad / "Zr "                → light grey
///   * steel / SS / vessel / barrel / iron    → darker grey
///   * concrete                               → tan
///   * control / RCCA / B₄C / B4C / AIC / absorber → yellow
///   * boron / boric / B10                    → yellow-orange
///   * lead-bismuth / LBE / PbBi              → dark teal-grey
///   * lead / Pb                              → dark slate
///   * graphite / carbon                      → near-black grey
///   * sodium / Na coolant                    → silver
///   * helium / CO₂ / gas                     → very light cyan
///   * polyethylene / CH₂ / paraffin          → pale pink
///   * air / void / vacuum                    → near-black
///   * reflector                              → green
///   * moderator                              → light blue
pub fn auto_color_from_name(name: &str) -> Option<[u8; 3]> {
    let n = name.to_lowercase();
    let any = |needles: &[&str]| needles.iter().any(|k| n.contains(k));

    if any(&["heavy water", "d2o", "d₂o"]) {
        return Some([40, 80, 180]);
    }
    if any(&["light water", "h2o", "h₂o"]) || (n.contains("water") && !n.contains("heavy")) {
        return Some([80, 150, 230]);
    }
    if any(&["mox", "plutonium", "puo2", "puo₂"]) {
        return Some([240, 140, 50]);
    }
    // Fuel-burnup keywords work even without an explicit "fuel"
    // word — reload patterns commonly say just "first / second /
    // third cycle" or "fresh / mid / burnt".
    let is_fuel_keyword =
        any(&["uo2", "uo₂", "uranium", "fuel", "first cycle", "1st cycle",
              "second cycle", "2nd cycle", "third cycle", "3rd cycle",
              "boc", "eoc", "fresh", "mid-core", "mid core"]);
    if is_fuel_keyword {
        if any(&["burnt", "depleted", "spent", "third", "3rd cycle", "eoc"]) {
            return Some([120, 60, 80]); // burnt — purple-red
        }
        if any(&["mid", "second", "2nd"]) {
            return Some([220, 120, 60]); // mid-cycle — orange
        }
        return Some([200, 80, 60]); // fresh — red
    }
    if any(&["zircaloy", "clad", "zr "]) || n == "zr" {
        return Some([170, 170, 170]);
    }
    if any(&["steel", "vessel", "barrel", "iron"]) || n.split_whitespace().any(|w| w == "ss") {
        return Some([110, 110, 120]);
    }
    if n.contains("concrete") {
        return Some([180, 160, 120]);
    }
    if any(&["control", "rcca", "b4c", "b₄c", "aic", "absorber"]) {
        return Some([255, 220, 60]);
    }
    if any(&["boron", "boric", "b10", "b-10"]) {
        return Some([255, 200, 80]);
    }
    if any(&["lead-bismuth", "lbe", "pbbi", "pb-bi"]) {
        return Some([60, 80, 100]);
    }
    if n.contains("lead")
        || n.split_whitespace().any(|w| w == "pb")
        || n.starts_with("pb ")
    {
        return Some([70, 70, 90]);
    }
    if any(&["graphite", "carbon"]) {
        return Some([60, 60, 60]);
    }
    if any(&["sodium", "na coolant"]) || n.split_whitespace().any(|w| w == "na") {
        return Some([200, 200, 220]);
    }
    if any(&["helium", "co2", "co₂", " gas"]) || n == "gas" {
        return Some([210, 240, 250]);
    }
    if any(&["polyethylene", "ch2", "ch₂", "paraffin"]) {
        return Some([240, 200, 210]);
    }
    if any(&["air", "void", "vacuum"]) {
        return Some([30, 30, 35]);
    }
    if n.contains("reflector") {
        return Some([60, 180, 100]);
    }
    if n.contains("moderator") {
        return Some([60, 180, 220]);
    }
    None
}

/// Build a legend from a list of materials and a palette. Material
/// at index `i` is paired with `palette.colors[i]` (or `palette.void`
/// for indices past the palette).
///
/// This is the OpenMC-Python-style auto-derivation: define your
/// materials once, the legend follows.
pub fn legend_from_materials<M: NamedMaterial>(
    materials: &[M],
    palette: &MaterialPalette,
) -> Vec<LegendEntry> {
    materials
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let color = palette.colors.get(i).copied().unwrap_or(palette.void);
            LegendEntry::new(m.name(), color)
        })
        .collect()
}

/// One-shot OpenMC-style preview: open a window, render the geometry
/// top-down, derive the legend from the supplied materials list.
/// Wraps [`show_window`] + [`render_top_down`] + [`legend_from_materials`].
///
/// Pass `palette = None` to auto-derive colours from each material's
/// name via [`MaterialPalette::for_materials`]: light water becomes
/// blue, fresh fuel red, MOX orange, lead dark, and so on. Pass an
/// explicit [`MaterialPalette`] to override.
pub fn preview_geometry<M: NamedMaterial>(
    initial: Viewport,
    title: &str,
    cells: &[Cell],
    surfaces: &[Surface],
    materials: &[M],
    cell_to_material: impl Fn(usize) -> usize + Sync,
    palette: Option<MaterialPalette>,
) {
    let palette = palette.unwrap_or_else(|| MaterialPalette::for_materials(materials));
    let legend = legend_from_materials(materials, &palette);
    show_window(initial, title, legend, |vp| {
        render_top_down(cells, surfaces, &cell_to_material, &palette, vp)
    });
}

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

impl MaterialPalette {
    /// Build a palette by looking up each material's name through
    /// [`auto_color_from_name`]. Materials whose names don't match
    /// any keyword fall back to the index-based [`Default`] palette,
    /// so the result still has a colour at every index.
    pub fn for_materials<M: NamedMaterial>(materials: &[M]) -> Self {
        let fallback = Self::default();
        let colors: Vec<[u8; 3]> = materials
            .iter()
            .enumerate()
            .map(|(i, m)| {
                auto_color_from_name(m.name())
                    .unwrap_or_else(|| {
                        fallback.colors.get(i).copied().unwrap_or(fallback.void)
                    })
            })
            .collect();
        Self {
            colors,
            void: fallback.void,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_color(c: Option<[u8; 3]>, expected: [u8; 3]) {
        assert_eq!(c, Some(expected));
    }

    #[test]
    fn light_water_is_cool_blue() {
        approx_color(auto_color_from_name("light water moderator"), [80, 150, 230]);
        approx_color(auto_color_from_name("H2O"), [80, 150, 230]);
        approx_color(auto_color_from_name("H₂O at 583 K"), [80, 150, 230]);
    }

    #[test]
    fn heavy_water_is_darker_blue() {
        approx_color(auto_color_from_name("heavy water"), [40, 80, 180]);
        approx_color(auto_color_from_name("D2O"), [40, 80, 180]);
        approx_color(auto_color_from_name("D₂O coolant"), [40, 80, 180]);
    }

    #[test]
    fn fuel_keywords_split_by_burnup() {
        approx_color(
            auto_color_from_name("UO₂ fuel (fresh, 3.7 % ²³⁵U)"),
            [200, 80, 60],
        );
        approx_color(
            auto_color_from_name("second cycle (mid-core)"),
            [220, 120, 60],
        );
        approx_color(
            auto_color_from_name("third cycle (periphery)"),
            [120, 60, 80],
        );
    }

    #[test]
    fn mox_distinct_from_uo2() {
        approx_color(auto_color_from_name("MOX fuel 7 % Pu"), [240, 140, 50]);
    }

    #[test]
    fn structural_materials() {
        approx_color(auto_color_from_name("Zircaloy-4 clad"), [170, 170, 170]);
        approx_color(
            auto_color_from_name("steel core barrel + pressure vessel"),
            [110, 110, 120],
        );
        approx_color(auto_color_from_name("biological concrete"), [180, 160, 120]);
    }

    #[test]
    fn absorbers_and_lead() {
        approx_color(
            auto_color_from_name("RCCA control cluster (B₄C / AIC)"),
            [255, 220, 60],
        );
        approx_color(auto_color_from_name("lead shielding"), [70, 70, 90]);
        approx_color(auto_color_from_name("Pb-Bi eutectic"), [60, 80, 100]);
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(auto_color_from_name("Unobtainium-235").is_none());
        assert!(auto_color_from_name("frobnicator").is_none());
    }
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
///     and a smaller window zooms *in*.
///   * Scroll wheel — multiplicative zoom around the viewport's
///     centre.
///   * `R` — reset to the initial viewport.
///   * `L` — toggle the legend popup (a second window listing the
///     palette colours and what they mean). Closing the legend
///     window manually has the same effect as toggling it off.
///   * `Esc` or main window-X — close.
///
/// Pass `legend` empty (`Vec::new()`) to disable the popup; in that
/// case `L` does nothing.
pub fn show_window<F>(
    initial: Viewport,
    title: &str,
    legend: Vec<LegendEntry>,
    mut render: F,
) where
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
    let mut prev_l_pressed = false;
    let mut legend_window: Option<(Window, Vec<u32>, usize, usize)> = None;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let cur_size = window.get_size();
        let mut needs_render = false;

        if cur_size != last_size && cur_size.0 > 0 && cur_size.1 > 0 {
            let cx = (viewport.x_min + viewport.x_max) * 0.5;
            let cy = (viewport.y_min + viewport.y_max) * 0.5;
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

        let r_now = window.is_key_down(Key::R);
        if r_now && !prev_r_pressed {
            viewport = initial;
            last_size = (0, 0);
            needs_render = true;
        }
        prev_r_pressed = r_now;

        // Legend toggle. Pressing L when the legend is closed opens
        // it; pressing again closes it. If the user closes the
        // legend window directly (X), drop it on the next iteration.
        let l_now = window.is_key_down(Key::L);
        if l_now && !prev_l_pressed && !legend.is_empty() {
            if legend_window.is_some() {
                legend_window = None;
            } else {
                let (buf, w, h) = render_legend(&legend);
                let win = Window::new(
                    "rust-mc-sim — legend",
                    w,
                    h,
                    WindowOptions {
                        resize: false,
                        ..WindowOptions::default()
                    },
                )
                .unwrap_or_else(|e| panic!("failed to open legend window: {e}"));
                legend_window = Some((win, buf, w, h));
            }
        }
        prev_l_pressed = l_now;

        if let Some((lw, lbuf, lw_w, lw_h)) = legend_window.as_mut() {
            if !lw.is_open() || lw.is_key_down(Key::Escape) {
                legend_window = None;
            } else {
                lw.update_with_buffer(lbuf, *lw_w, *lw_h)
                    .unwrap_or_else(|e| panic!("failed to blit legend: {e}"));
            }
        }

        if needs_render {
            buffer = render(&viewport);
        }
        window
            .update_with_buffer(&buffer, viewport.width as usize, viewport.height as usize)
            .unwrap_or_else(|e| panic!("failed to blit framebuffer: {e}"));
    }
}

/// Build the legend popup framebuffer. Each entry gets one row:
/// 32 × 32 colour swatch on the left, label rendered with `font8x8`
/// to the right. Window size is sized to fit all entries.
fn render_legend(legend: &[LegendEntry]) -> (Vec<u32>, usize, usize) {
    use font8x8::UnicodeFonts;

    let scale = 2_usize; // upscale 8 × 8 glyphs to 16 × 16 for readability
    let row_height = 36_usize;
    let swatch = 28_usize;
    let pad_x = 12_usize;
    let pad_y = 8_usize;
    let max_label_chars = legend.iter().map(|e| e.label.chars().count()).max().unwrap_or(0);
    let label_width = max_label_chars * 8 * scale;
    let w = pad_x + swatch + 12 + label_width + pad_x;
    let h = pad_y * 2 + row_height * legend.len().max(1);
    let bg: u32 = 0x1A1A24;
    let text_color: u32 = 0xE8E8F0;
    let mut buf = vec![bg; w * h];

    for (row, entry) in legend.iter().enumerate() {
        let y0 = pad_y + row * row_height + (row_height - swatch) / 2;
        let x0 = pad_x;
        let swatch_color = pack_rgb(entry.color);
        // Draw the swatch with a 1-px frame for contrast.
        for dy in 0..swatch {
            for dx in 0..swatch {
                let py = y0 + dy;
                let px = x0 + dx;
                if px < w && py < h {
                    let on_border =
                        dx == 0 || dy == 0 || dx == swatch - 1 || dy == swatch - 1;
                    buf[py * w + px] = if on_border { 0x000000 } else { swatch_color };
                }
            }
        }
        // Draw the label.
        let label_x = x0 + swatch + 12;
        let label_y = pad_y + row * row_height + (row_height - 8 * scale) / 2;
        for (i, ch) in entry.label.chars().enumerate() {
            let glyph = font8x8::BASIC_FONTS.get(ch).unwrap_or([0u8; 8]);
            for (gy, byte) in glyph.iter().enumerate() {
                for gx in 0..8 {
                    if (byte >> gx) & 1 != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = label_x + i * 8 * scale + gx * scale + sx;
                                let py = label_y + gy * scale + sy;
                                if px < w && py < h {
                                    buf[py * w + px] = text_color;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (buf, w, h)
}
