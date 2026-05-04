//! French CP1-class PWR core preview — 900 MWe Westinghouse 3-loop
//! design (Almaraz, Ascó, Tricastin, Bugey, Blayais, Cruas, …).
//!
//! Layout: 157 fuel assemblies on a 21.5 cm pitch arranged in the
//! standard CP1 footprint inside a steel core barrel, with a water
//! reflector annulus and a steel pressure vessel. Each assembly is
//! drawn as a circle and colour-coded by 3-batch reload region:
//!
//!   * red    — first cycle (fresh fuel, highest enrichment)
//!   * orange — second cycle (once-burnt, mid-core)
//!   * green  — third cycle (twice-burnt, periphery)
//!   * yellow — control-rod cluster positions
//!
//! Pin-level structure inside an assembly (pellet + clad + water on
//! a 17 × 17 grid) is the same as `examples/10_pwr_pin_cell.rs` —
//! the whole-core preview here intentionally renders one assembly
//! per circle so the macro pattern reads cleanly without having to
//! sample 45 000 pin cells per pixel.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features preview --example 14_french_cp1_core_preview
//! ```

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, between, inside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::Surface;
use rust_mc_sim::preview::{MaterialPalette, Viewport, render_top_down, show_window};

// ── Plant geometry (cm) ─────────────────────────────────────────────
// CP1 fuel assembly side: 21.504 cm. Reactor pressure vessel inner
// radius ≈ 199 cm. Numbers are approximate, dimensioned for a clean
// preview at this scale.
const ASSEMBLY_PITCH: f64 = 21.5;
const ASSEMBLY_RADIUS: f64 = 10.0; // visual radius of an assembly disk
const CORE_BARREL_INNER: f64 = 165.0;
const CORE_BARREL_OUTER: f64 = 170.0;
const REFLECTOR_OUTER: f64 = 195.0;
const VESSEL_OUTER: f64 = 220.0;

// ── 3-batch reload colour zones (radial cycle) ──────────────────────
// Core radius dividing the three batches. CP1 uses out-in-in
// (Westinghouse low-leakage) but for visual clarity we draw the
// rings as concentric.
const RADIUS_BATCH_1: f64 = 60.0;  // inner — fresh fuel
const RADIUS_BATCH_2: f64 = 120.0; // mid   — once-burnt

// ── Control-rod cluster grid positions (i, j on the lattice) ────────
// CP1 has 53 RCCAs total; we draw a representative subset of the
// AIC + B4C cluster pattern to keep the preview readable.
const CONTROL_POSITIONS: &[(i32, i32)] = &[
    (0, 0),  // central
    (0, 4), (0, -4), (4, 0), (-4, 0),
    (4, 4), (4, -4), (-4, 4), (-4, -4),
    (0, 7), (0, -7), (7, 0), (-7, 0),
    (3, 6), (3, -6), (-3, 6), (-3, -6),
    (6, 3), (6, -3), (-6, 3), (-6, -3),
];

// ── Material indices into the palette ───────────────────────────────
const MAT_FUEL_FRESH: usize = 0;     // red
const MAT_FUEL_MID: usize = 4;       // green (we reuse "moderator" slot)
const MAT_FUEL_BURNT: usize = 5;     // purple-ish ("instrumentation")
const MAT_WATER: usize = 2;
const MAT_CONTROL: usize = 3;
const MAT_STEEL: usize = 6;

fn main() {
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut cell_materials: Vec<usize> = Vec::new();

    // 1) Vessel + barrel cylinders.
    let s_barrel_inner = push_surface(
        &mut surfaces,
        cyl(0.0, 0.0, CORE_BARREL_INNER, BoundaryCondition::Transmission),
    );
    let s_barrel_outer = push_surface(
        &mut surfaces,
        cyl(0.0, 0.0, CORE_BARREL_OUTER, BoundaryCondition::Transmission),
    );
    let s_reflector_outer = push_surface(
        &mut surfaces,
        cyl(0.0, 0.0, REFLECTOR_OUTER, BoundaryCondition::Transmission),
    );
    let s_vessel_outer = push_surface(
        &mut surfaces,
        cyl(0.0, 0.0, VESSEL_OUTER, BoundaryCondition::Vacuum),
    );

    // 2) Place 157 assemblies on the CP1 lattice. Iterate over a
    // 17 × 17 grid and accept those whose centre falls inside the
    // CP1 footprint, which we approximate as the maximum circle that
    // fits an integer count of assemblies on the lattice.
    let cp1_radius = 7.6 * ASSEMBLY_PITCH;
    let mut assembly_count = 0;
    for j in -8_i32..=8 {
        for i in -8_i32..=8 {
            let cx = (i as f64) * ASSEMBLY_PITCH;
            let cy = (j as f64) * ASSEMBLY_PITCH;
            let r = (cx * cx + cy * cy).sqrt();
            if r > cp1_radius - ASSEMBLY_PITCH * 0.5 {
                continue;
            }
            assembly_count += 1;
            let mat = classify_assembly(i, j, r);
            let s = push_surface(
                &mut surfaces,
                cyl(cx, cy, ASSEMBLY_RADIUS, BoundaryCondition::Transmission),
            );
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                inside(s),
                CellFill::Material(mat as u32),
            ));
            cell_materials.push(mat);
        }
    }

    // 3) Water inside the core barrel — anywhere not already an
    // assembly disk. We add one cell with region `inside(barrel)`;
    // the assembly cells appear earlier in the cell list and so
    // shadow this water cell wherever they cover.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        inside(s_barrel_inner),
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // 4) Steel core barrel.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_inner, s_barrel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    // 5) Water reflector annulus between barrel and reflector outer.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_outer, s_reflector_outer),
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // 6) Steel pressure vessel.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_reflector_outer, s_vessel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    let _ = s_vessel_outer;

    // 7) Render and show. The closure re-renders on resize / scroll /
    // 'R' reset; first frame is the initial 1000 × 1000 view.
    let viewport = Viewport::square_centered(VESSEL_OUTER + 5.0, 0.0, 1000);
    let palette = MaterialPalette::default();
    println!(
        "CP1-class core: {assembly_count} assemblies, {} surfaces, {} cells",
        surfaces.len(),
        cells.len()
    );
    println!(
        "drag the window to zoom, scroll to zoom around centre, R to reset, Esc to close."
    );
    show_window(
        viewport,
        "rust-mc-sim — French CP1 900 MWe core preview",
        |vp| {
            let t0 = std::time::Instant::now();
            let buf = render_top_down(
                &cells,
                &surfaces,
                |cell_idx| cell_materials[cell_idx],
                &palette,
                vp,
            );
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "  rendered {}×{} px in {dt:.2} s ({:.1} Mpx/s)",
                vp.width,
                vp.height,
                buf.len() as f64 * 1.0e-6 / dt.max(1.0e-9)
            );
            buf
        },
    );
}

fn classify_assembly(i: i32, j: i32, radius_cm: f64) -> usize {
    if CONTROL_POSITIONS.contains(&(i, j)) {
        MAT_CONTROL
    } else if radius_cm < RADIUS_BATCH_1 {
        MAT_FUEL_FRESH
    } else if radius_cm < RADIUS_BATCH_2 {
        MAT_FUEL_MID
    } else {
        MAT_FUEL_BURNT
    }
}

fn cyl(cx: f64, cy: f64, r: f64, bc: BoundaryCondition) -> Surface {
    Surface::CylinderZ {
        center_x: cx,
        center_y: cy,
        radius: r,
        bc,
    }
}

fn push_surface(surfaces: &mut Vec<Surface>, surface: Surface) -> usize {
    let idx = surfaces.len();
    surfaces.push(surface);
    idx
}
