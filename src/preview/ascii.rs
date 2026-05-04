//! Shell-only ASCII renderer — prints a top-down CSG slice as a
//! grid of characters. No window manager, no PNG, no GUI deps; just
//! `println!`. Useful as a fast diagnostic when you want to verify
//! that a geometry build is correct before opening a window or
//! launching a Monte Carlo run.
//!
//! The default character map picks a glyph based on each material's
//! name via the same name-keyword heuristic [`auto_color_from_name`]
//! uses for colours; pass an explicit `Vec<char>` to override.

use crate::geometry::bvh::Bvh;
use crate::geometry::{Cell, Surface, Vec3, ray};

use super::render::{NamedMaterial, Viewport, auto_color_from_name};

/// Pick a glyph from a material name using the same keyword tree as
/// [`super::auto_color_from_name`]. `' '` for unknown / void.
pub fn ascii_glyph_for_name(name: &str) -> char {
    let n = name.to_lowercase();
    let any = |needles: &[&str]| needles.iter().any(|k| n.contains(k));
    if any(&["heavy water", "d2o", "d₂o"]) {
        return 'D';
    }
    if any(&["light water", "h2o", "h₂o"]) || (n.contains("water") && !n.contains("heavy")) {
        return '.';
    }
    if any(&["mox", "plutonium", "puo2", "puo₂"]) {
        return 'M';
    }
    let is_fuel = any(&[
        "uo2", "uo₂", "uranium", "fuel", "first cycle", "1st cycle",
        "second cycle", "2nd cycle", "third cycle", "3rd cycle",
        "boc", "eoc", "fresh", "mid-core", "mid core",
    ]);
    if is_fuel {
        if any(&["burnt", "depleted", "spent", "third", "3rd cycle", "eoc"]) {
            return 'b';
        }
        if any(&["mid", "second", "2nd"]) {
            return 'm';
        }
        return '#';
    }
    if any(&["zircaloy", "clad", "zr "]) || n == "zr" {
        return '+';
    }
    if any(&["steel", "vessel", "barrel", "iron"])
        || n.split_whitespace().any(|w| w == "ss")
    {
        return '=';
    }
    if n.contains("concrete") {
        return ':';
    }
    if any(&["control", "rcca", "b4c", "b₄c", "aic", "absorber"]) {
        return 'X';
    }
    if any(&["lead-bismuth", "lbe", "pbbi"]) {
        return 'p';
    }
    if n.contains("lead") {
        return 'P';
    }
    if any(&["graphite", "carbon"]) {
        return 'C';
    }
    if any(&["sodium"]) || n.split_whitespace().any(|w| w == "na") {
        return '~';
    }
    if any(&["air", "void", "vacuum"]) {
        return ' ';
    }
    if n.contains("reflector") {
        return 'r';
    }
    if n.contains("moderator") {
        return ',';
    }
    '?'
}

/// Render a top-down ASCII slice and return it as one string. Each
/// cell's character is taken from `glyphs[i]`; pass `None` to derive
/// glyphs from material names.
pub fn render_ascii<M: NamedMaterial>(
    cells: &[Cell],
    surfaces: &[Surface],
    materials: &[M],
    cell_to_material: impl Fn(usize) -> usize,
    glyphs: Option<Vec<char>>,
    viewport: &Viewport,
) -> String {
    let glyphs: Vec<char> = glyphs.unwrap_or_else(|| {
        materials.iter().map(|m| ascii_glyph_for_name(m.name())).collect()
    });
    let bvh = Bvh::build(cells);
    let dx = (viewport.x_max - viewport.x_min) / viewport.width as f64;
    let dy = (viewport.y_max - viewport.y_min) / viewport.height as f64;
    let mut out = String::with_capacity(((viewport.width + 1) * viewport.height) as usize);

    for py in 0..viewport.height {
        let world_y = viewport.y_max - (py as f64 + 0.5) * dy;
        for px in 0..viewport.width {
            let world_x = viewport.x_min + (px as f64 + 0.5) * dx;
            let pos = Vec3::new(world_x, world_y, viewport.z_slice);
            let ch = match ray::find_cell_bvh(pos, surfaces, cells, &bvh) {
                Some(idx) => {
                    let mat = cell_to_material(idx);
                    glyphs.get(mat).copied().unwrap_or(' ')
                }
                None => ' ',
            };
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

/// Print a top-down ASCII slice to stdout with a small legend.
/// Each cell character is rendered against an ANSI 24-bit background
/// colour derived from its material name (the same palette
/// [`super::auto_color_from_name`] uses for the windowed
/// previewer), so water reads as a blue field, fuel red, control
/// rods yellow, etc — the terminal looks like a real reactor map.
///
/// On terminals without 24-bit colour (Windows < 10 cmd.exe pre
/// VT100, dumb pipes) the escape codes pass through visibly. Pipe
/// to a file or pass `colour: false` to [`print_ascii_with`] for
/// plain output.
pub fn print_ascii<M: NamedMaterial>(
    cells: &[Cell],
    surfaces: &[Surface],
    materials: &[M],
    cell_to_material: impl Fn(usize) -> usize,
    viewport: &Viewport,
) {
    print_ascii_with(cells, surfaces, materials, cell_to_material, viewport, true);
}

/// Variant of [`print_ascii`] that lets you disable the ANSI colour
/// path for plain-text output.
pub fn print_ascii_with<M: NamedMaterial>(
    cells: &[Cell],
    surfaces: &[Surface],
    materials: &[M],
    cell_to_material: impl Fn(usize) -> usize,
    viewport: &Viewport,
    colour: bool,
) {
    let glyphs: Vec<char> = materials
        .iter()
        .map(|m| ascii_glyph_for_name(m.name()))
        .collect();
    let colours: Vec<Option<[u8; 3]>> = materials
        .iter()
        .map(|m| auto_color_from_name(m.name()))
        .collect();

    let bvh = Bvh::build(cells);
    let dx = (viewport.x_max - viewport.x_min) / viewport.width as f64;
    let dy = (viewport.y_max - viewport.y_min) / viewport.height as f64;
    let mut last_color: Option<[u8; 3]> = None;
    let mut out = String::new();

    for py in 0..viewport.height {
        let world_y = viewport.y_max - (py as f64 + 0.5) * dy;
        for px in 0..viewport.width {
            let world_x = viewport.x_min + (px as f64 + 0.5) * dx;
            let pos = Vec3::new(world_x, world_y, viewport.z_slice);
            let (ch, this_color) = match ray::find_cell_bvh(pos, surfaces, cells, &bvh) {
                Some(idx) => {
                    let mat = cell_to_material(idx);
                    let g = glyphs.get(mat).copied().unwrap_or(' ');
                    let c = colours.get(mat).copied().flatten();
                    (g, c)
                }
                None => (' ', None),
            };
            if colour {
                if this_color != last_color {
                    if let Some([r, g, b]) = this_color {
                        out.push_str(&format!("\x1b[48;2;{};{};{}m", r, g, b));
                    } else {
                        out.push_str("\x1b[49m"); // reset background
                    }
                    last_color = this_color;
                }
            }
            out.push(ch);
        }
        // Reset at end of each row so terminal redraws don't smear
        // background colour into the right margin.
        if colour {
            out.push_str("\x1b[0m");
            last_color = None;
        }
        out.push('\n');
    }
    print!("{out}");
    println!("legend:");
    for (m, g) in materials.iter().zip(glyphs.iter()) {
        if colour {
            if let Some([r, gn, b]) = auto_color_from_name(m.name()) {
                println!(
                    "  \x1b[48;2;{r};{gn};{b}m {g} \x1b[0m  {}",
                    m.name()
                );
                continue;
            }
        }
        println!("  '{g}'  {}", m.name());
    }
}
