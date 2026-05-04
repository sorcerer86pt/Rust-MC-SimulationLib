//! French CP1-class PWR whole-core preview — 900 MWe Westinghouse
//! 3-loop design (Almaraz / Ascó / Tricastin / Bugey / Blayais /
//! Cruas …). 161 fuel assemblies arranged on a 21.5 cm pitch in the
//! standard CP1 footprint, surrounded by core barrel, water
//! reflector and pressure vessel.
//!
//! Materials are the simulation's ground-truth list — the same
//! `transport::material::Material` type the eigenvalue and
//! fixed-source drivers consume. The preview's legend is auto-built
//! from this list via [`preview_geometry`], so renaming a material
//! or adding a new one shows up in both the colour swatches and the
//! pop-up legend without any further glue.
//!
//! Hot keys: drag-resize / scroll zoom / R reset / L legend / Esc.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features preview --example 14_french_cp1_core_preview
//! ```

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, between, inside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::Surface;
use rust_mc_sim::preview::{Viewport, preview_geometry};
use rust_mc_sim::transport::material::Material;

// ── Plant geometry (cm) ─────────────────────────────────────────────
const ASSEMBLY_PITCH: f64 = 21.5;
const ASSEMBLY_RADIUS: f64 = 10.0;
const CORE_BARREL_INNER: f64 = 165.0;
const CORE_BARREL_OUTER: f64 = 170.0;
const REFLECTOR_OUTER: f64 = 195.0;
const VESSEL_OUTER: f64 = 220.0;

// 3-batch reload colour zones (radial, simplified).
const RADIUS_BATCH_1: f64 = 60.0;
const RADIUS_BATCH_2: f64 = 120.0;

// Representative subset of the 53 RCCA cluster positions.
const CONTROL_POSITIONS: &[(i32, i32)] = &[
    (0, 0), (0, 4), (0, -4), (4, 0), (-4, 0),
    (4, 4), (4, -4), (-4, 4), (-4, -4),
    (0, 7), (0, -7), (7, 0), (-7, 0),
    (3, 6), (3, -6), (-3, 6), (-3, -6),
    (6, 3), (6, -3), (-6, 3), (-6, -3),
];

// Material indices into the materials list below.
const MAT_FUEL_FRESH: usize = 0;
const MAT_FUEL_MID: usize = 1;
const MAT_FUEL_BURNT: usize = 2;
const MAT_CONTROL: usize = 3;
const MAT_WATER: usize = 4;
const MAT_STEEL: usize = 5;

fn main() {
    // Materials list — names propagate to the legend automatically.
    // For a real CP1 core the three "burnup" materials would carry
    // different nuclide vectors with different ²³⁵U / Pu / FP
    // inventories at BOC of the cycle; here we just give them
    // labels.
    let materials: Vec<Material> = vec![
        Material::new("first cycle (fresh, ~3.7 % ²³⁵U)", 900.0),
        Material::new("second cycle (mid-core)", 900.0),
        Material::new("third cycle (periphery)", 900.0),
        Material::new("RCCA control cluster (B₄C / AIC)", 583.0),
        Material::new("light water reflector", 583.0),
        Material::new("steel core barrel + pressure vessel", 583.0),
    ];

    let mut surfaces: Vec<Surface> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut cell_materials: Vec<usize> = Vec::new();

    let s_barrel_inner = push(&mut surfaces, cyl(0.0, 0.0, CORE_BARREL_INNER));
    let s_barrel_outer = push(&mut surfaces, cyl(0.0, 0.0, CORE_BARREL_OUTER));
    let s_reflector_outer = push(&mut surfaces, cyl(0.0, 0.0, REFLECTOR_OUTER));
    let s_vessel_outer = push(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: VESSEL_OUTER,
            bc: BoundaryCondition::Vacuum,
        },
    );
    let _ = s_vessel_outer;

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
            let s = push(&mut surfaces, cyl(cx, cy, ASSEMBLY_RADIUS));
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                inside(s),
                CellFill::Material(mat as u32),
            ));
            cell_materials.push(mat);
        }
    }

    // Water inside core barrel; assemblies (defined earlier) shadow
    // this cell wherever they cover.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        inside(s_barrel_inner),
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // Steel barrel.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_inner, s_barrel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    // Water reflector annulus.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_outer, s_reflector_outer),
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // Pressure vessel.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_reflector_outer, s_vessel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    let viewport = Viewport::square_centered(VESSEL_OUTER * 1.25, 0.0, 1000);
    println!(
        "CP1-class core: {assembly_count} assemblies, {} surfaces, {} cells, {} materials.",
        surfaces.len(),
        cells.len(),
        materials.len()
    );
    println!(
        "drag/scroll to zoom, R to reset, L for legend, Esc to close."
    );
    preview_geometry(
        viewport,
        "rust-mc-sim — French CP1 900 MWe core preview",
        &cells,
        &surfaces,
        &materials,
        |cell_idx| cell_materials[cell_idx],
        None, // auto-derive colours from material names
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

fn cyl(cx: f64, cy: f64, r: f64) -> Surface {
    Surface::CylinderZ {
        center_x: cx,
        center_y: cy,
        radius: r,
        bc: BoundaryCondition::Transmission,
    }
}

fn push(surfaces: &mut Vec<Surface>, surface: Surface) -> usize {
    let idx = surfaces.len();
    surfaces.push(surface);
    idx
}
