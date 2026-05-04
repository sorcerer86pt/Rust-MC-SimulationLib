//! Geometry previewer demo: builds a stylised PWR cross-section —
//! pressure vessel, core barrel, water reflector, a 17 × 17 fuel
//! assembly with four control-rod positions — and opens a window
//! showing the top-down slice. Press Esc or click the window's X to
//! close.
//!
//! No simulation is run; this is purely the geometry layer, used to
//! sanity-check a build before launching a Monte Carlo job.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features preview --example 13_reactor_preview
//! ```

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, Region, between, inside, outside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::{Surface, Vec3};
use rust_mc_sim::preview::{MaterialPalette, Viewport, render_top_down, show_window};

// ── Materials ───────────────────────────────────────────────────────
// Indices are referenced by `cell_to_material` and looked up in the
// palette below.
const MAT_FUEL: usize = 0;
const MAT_CLAD: usize = 1;
const MAT_WATER: usize = 2;
const MAT_CONTROL: usize = 3;
const MAT_STEEL: usize = 6;

// ── Pin / lattice geometry ──────────────────────────────────────────
const PITCH: f64 = 1.260;
const PELLET_R: f64 = 0.4096;
const CLAD_OR: f64 = 0.4750;
const N_PIN_SIDE: i32 = 17;
const ASSEMBLY_HALF: f64 = (N_PIN_SIDE as f64) * PITCH * 0.5; // ≈ 10.71 cm

// ── Vessel / barrel — concentric cylinders centred at origin ────────
const CORE_BARREL_INNER: f64 = 13.0;
const CORE_BARREL_OUTER: f64 = 14.5;
const REFLECTOR_OUTER: f64 = 18.0;
const VESSEL_OUTER: f64 = 21.0;

// Control-rod pin coordinates (i, j) inside the 17 × 17 lattice.
// Real PWR assemblies have 24 guide tubes + 1 instrumentation; here
// we light up four representative positions for visual clarity.
const CONTROL_ROD_POSITIONS: &[(i32, i32)] = &[(4, 4), (4, 12), (12, 4), (12, 12)];

fn main() {
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut cell_materials: Vec<usize> = Vec::new();

    // 1) Concentric vessel + barrel cylinders. Surfaces s0..s3.
    let s_barrel_inner = push_surface(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CORE_BARREL_INNER,
            bc: BoundaryCondition::Transmission,
        },
    );
    let s_barrel_outer = push_surface(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CORE_BARREL_OUTER,
            bc: BoundaryCondition::Transmission,
        },
    );
    let s_reflector_outer = push_surface(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: REFLECTOR_OUTER,
            bc: BoundaryCondition::Transmission,
        },
    );
    let s_vessel_outer = push_surface(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: VESSEL_OUTER,
            bc: BoundaryCondition::Vacuum,
        },
    );
    // Bounding box around the full picture so cells outside the
    // vessel evaluate as void via `find_cell` returning None.
    // (We don't need an explicit air cell for the preview.)

    // 2) Fuel pins inside a 17 × 17 lattice. Each pin contributes
    // two cylinder surfaces (pellet OR + clad OR) and three cells
    // (fuel, clad, none — water cell is one big global cell below).
    let mut pin_surfaces: Vec<(usize, usize)> = Vec::new();
    for j in 0..N_PIN_SIDE {
        for i in 0..N_PIN_SIDE {
            let cx = (i as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let cy = (j as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let s_pellet = push_surface(
                &mut surfaces,
                Surface::CylinderZ {
                    center_x: cx,
                    center_y: cy,
                    radius: PELLET_R,
                    bc: BoundaryCondition::Transmission,
                },
            );
            let s_clad = push_surface(
                &mut surfaces,
                Surface::CylinderZ {
                    center_x: cx,
                    center_y: cy,
                    radius: CLAD_OR,
                    bc: BoundaryCondition::Transmission,
                },
            );
            let is_control = CONTROL_ROD_POSITIONS.contains(&(i, j));
            // Pellet cell. Material: control (yellow) for guide-tube
            // positions, fuel (red) elsewhere.
            let pellet_mat = if is_control { MAT_CONTROL } else { MAT_FUEL };
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                inside(s_pellet),
                CellFill::Material(pellet_mat as u32),
            ));
            cell_materials.push(pellet_mat);
            // Clad annulus.
            cells.push(Cell::new(
                CellId(cells.len() as u32),
                between(s_pellet, s_clad),
                CellFill::Material(MAT_CLAD as u32),
            ));
            cell_materials.push(MAT_CLAD);
            pin_surfaces.push((s_pellet, s_clad));
        }
    }

    // 3) Water inside the assembly footprint (the cells between pins
    // and the assembly outer box). We approximate by saying "inside
    // core barrel AND outside every pin's clad". The CSG path
    // evaluates this as a chain of intersections / complements; with
    // 17 × 17 = 289 pins it's a tall expression but still O(N) per
    // pixel.
    let water_assembly = build_water_in_assembly(&pin_surfaces, s_barrel_inner);
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        water_assembly,
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // 4) Steel barrel annulus.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_inner, s_barrel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    // 5) Water reflector between core barrel outer and reflector outer.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_barrel_outer, s_reflector_outer),
        CellFill::Material(MAT_WATER as u32),
    ));
    cell_materials.push(MAT_WATER);

    // 6) Steel pressure vessel between vessel inner and outer.
    cells.push(Cell::new(
        CellId(cells.len() as u32),
        between(s_reflector_outer, s_vessel_outer),
        CellFill::Material(MAT_STEEL as u32),
    ));
    cell_materials.push(MAT_STEEL);

    let _ = (s_vessel_outer, ASSEMBLY_HALF); // shut up unused-warn

    // 7) Render and show. The closure is re-invoked on every
    // resize / scroll-zoom / `R` reset.
    // 25 % margin so non-square window resizes keep the vessel ring
    // fully on-screen.
    let viewport = Viewport::square_centered(VESSEL_OUTER * 1.25, 0.0, 900);
    let palette = MaterialPalette::default();
    println!(
        "{} surfaces, {} cells. drag the window to zoom, scroll to zoom around centre, R to reset, Esc to close.",
        surfaces.len(),
        cells.len()
    );
    show_window(
        viewport,
        "rust-mc-sim — PWR cross-section preview",
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

fn push_surface(surfaces: &mut Vec<Surface>, surface: Surface) -> usize {
    let idx = surfaces.len();
    surfaces.push(surface);
    idx
}

/// Region that's "inside the core barrel AND outside every pin clad".
/// Built as a left-folded chain of intersections + complements.
fn build_water_in_assembly(
    pin_surfaces: &[(usize, usize)],
    s_barrel_inner: usize,
) -> Region {
    let mut region = inside(s_barrel_inner);
    for &(_, s_clad) in pin_surfaces {
        region = Region::Intersection(Box::new(region), Box::new(outside(s_clad)));
    }
    region
}

// Silence `Vec3` unused warning — we don't use it in this file but
// some palette tweaks pull it in via `geometry`.
#[allow(dead_code)]
fn _unused(_: Vec3) {}
