//! Geometry previewer demo — a stylised PWR fuel-assembly cross
//! section (CP1 / Almaraz pin geometry, 17 × 17 lattice with four
//! control-rod positions) inside a circular vessel. Opens a window
//! showing the top-down slice. Hot keys: drag-resize / scroll
//! zoom / R reset / L legend / Esc close.
//!
//! The legend window is auto-built from the `Material` list, the
//! same way OpenMC's Python plotter derives legend entries from
//! `openmc.Materials`. Add or remove a material here and both the
//! legend and the colour mapping in the render follow it.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features preview --example 13_reactor_preview
//! ```

use std::sync::Arc;

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, Region, between, inside, outside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::Surface;
use rust_mc_sim::preview::{Viewport, preview_geometry};
use rust_mc_sim::transport::material::{Material, Nuclide};

// ── Material indices into the user's material list. The previewer
//    looks each cell's material index up here, then uses palette
//    index `i` for material `i`. Same indexing OpenMC uses.
const MAT_FUEL: usize = 0;
const MAT_CLAD: usize = 1;
const MAT_WATER: usize = 2;
const MAT_CONTROL: usize = 3;
const MAT_STEEL: usize = 4;

// Pin / lattice geometry (CP1 / Almaraz fresh fuel, see
// examples/10_pwr_pin_cell.rs for the simulation that uses these).
const PITCH: f64 = 1.260;
const PELLET_R: f64 = 0.4096;
const CLAD_OR: f64 = 0.4750;
const N_PIN_SIDE: i32 = 17;

// Vessel / barrel — concentric cylinders centred at origin.
const CORE_BARREL_INNER: f64 = 13.0;
const CORE_BARREL_OUTER: f64 = 14.5;
const REFLECTOR_OUTER: f64 = 18.0;
const VESSEL_OUTER: f64 = 21.0;

const CONTROL_ROD_POSITIONS: &[(i32, i32)] = &[(4, 4), (4, 12), (12, 4), (12, 12)];

fn main() {
    // Build the materials list with names. Composition is left empty
    // for the preview — the previewer only reads `name`. Real sims
    // populate the nuclide vectors via `nuclear::loader`.
    let materials: Vec<Material> = vec![
        Material::new("UO₂ fuel (fresh, 3.7 % ²³⁵U)", 900.0),
        Material::new("Zircaloy-4 clad", 600.0),
        Material::new("light water moderator", 583.0),
        Material::new("control rod / guide tube (B₄C / AIC)", 583.0),
        Material::new("steel core barrel + pressure vessel", 583.0),
    ];
    let _ = Arc::new(Nuclide::empty("placeholder", 1.0)); // silence unused

    // Geometry build.
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

    // 17 × 17 fuel-pin lattice.
    let mut pin_clads: Vec<usize> = Vec::new();
    for j in 0..N_PIN_SIDE {
        for i in 0..N_PIN_SIDE {
            let cx = (i as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let cy = (j as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let s_pellet = push(&mut surfaces, cyl(cx, cy, PELLET_R));
            let s_clad = push(&mut surfaces, cyl(cx, cy, CLAD_OR));
            pin_clads.push(s_clad);
            let pellet_mat = if CONTROL_ROD_POSITIONS.contains(&(i, j)) {
                MAT_CONTROL
            } else {
                MAT_FUEL
            };
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                inside(s_pellet),
                CellFill::Material(pellet_mat as u32),
            ));
            cell_materials.push(pellet_mat);
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                between(s_pellet, s_clad),
                CellFill::Material(MAT_CLAD as u32),
            ));
            cell_materials.push(MAT_CLAD);
        }
    }

    // Water inside the assembly: inside barrel ∩ outside every clad.
    let mut water_region = inside(s_barrel_inner);
    for &s_clad in &pin_clads {
        water_region = Region::Intersection(Box::new(water_region), Box::new(outside(s_clad)));
    }
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        water_region,
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

    // Reflector annulus.
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

    let viewport = Viewport::square_centered(VESSEL_OUTER * 1.25, 0.0, 900);
    println!(
        "{} surfaces, {} cells, {} materials. drag/scroll to zoom, R to reset, L for legend, Esc to close.",
        surfaces.len(),
        cells.len(),
        materials.len()
    );
    preview_geometry(
        viewport,
        "rust-mc-sim — PWR cross-section preview",
        &cells,
        &surfaces,
        &materials,
        |cell_idx| cell_materials[cell_idx],
        None, // auto-derive colours from material names
    );
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
