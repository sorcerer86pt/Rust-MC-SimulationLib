#![allow(clippy::unwrap_used, clippy::expect_used)]
//! ASCII geometry previews — same CSG geometry as the windowed
//! examples 13 and 14, but printed straight to the terminal. No GUI
//! deps needed at runtime, useful as a quick visual sanity check
//! over SSH / inside CI.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features preview --example 15_ascii_reactor_preview
//! ```

use rust_mc_sim::geometry::Surface;
use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, between, inside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::preview::{Viewport, print_ascii};
use rust_mc_sim::transport::material::Material;

fn main() {
    print_pin_cell();
    println!();
    print_cp1_core();
}

// ── Single PWR pin cell (CP1 / Almaraz dimensions) ──────────────────
fn print_pin_cell() {
    println!("=== Single PWR pin cell — 1.260 cm pitch ===\n");

    let materials = vec![
        Material::new("UO₂ fuel (fresh, 3.7 % ²³⁵U)", 900.0),
        Material::new("Zircaloy-4 clad", 600.0),
        Material::new("light water moderator", 583.0),
    ];
    let surfaces = vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 0.4096,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 0.4750,
            bc: BoundaryCondition::Transmission,
        },
    ];
    // Cells are assembled from the proper surface set in
    // `build_pin_cell` (which uses pitch planes for the moderator).
    let _ = surfaces;
    let cells = build_pin_cell();
    let cell_materials = vec![0_usize, 1, 2];
    let viewport = Viewport {
        x_min: -0.65,
        x_max: 0.65,
        y_min: -0.65,
        y_max: 0.65,
        z_slice: 0.0,
        width: 50,
        height: 25,
    };
    print_ascii(
        &cells,
        &surfaces_pin_cell(),
        &materials,
        |i| cell_materials[i],
        &viewport,
    );
}

fn surfaces_pin_cell() -> Vec<Surface> {
    vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 0.4096,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 0.4750,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: -0.63,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneX {
            x0: 0.63,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: -0.63,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: 0.63,
            bc: BoundaryCondition::Reflective,
        },
    ]
}

fn build_pin_cell() -> Vec<Cell> {
    use rust_mc_sim::geometry::cell::Region;
    let surfaces = surfaces_pin_cell();
    let inside_box = Region::Intersection(Box::new(between(2, 3)), Box::new(between(4, 5)));
    let outside_clad = Region::Intersection(
        Box::new(rust_mc_sim::geometry::cell::outside(1)),
        Box::new(inside_box),
    );
    vec![
        Cell::new(CellId(0), inside(0), CellFill::Material(0)).with_aabb_from_region(&surfaces),
        Cell::new(CellId(1), between(0, 1), CellFill::Material(1)).with_aabb_from_region(&surfaces),
        Cell::new(CellId(2), outside_clad, CellFill::Material(2)).with_aabb_from_region(&surfaces),
    ]
}

// ── CP1 whole-core layout, ASCII ───────────────────────────────────
fn print_cp1_core() {
    println!("=== French CP1 900 MWe core layout ===\n");

    const ASSEMBLY_PITCH: f64 = 21.5;
    const ASSEMBLY_RADIUS: f64 = 10.0;
    const CORE_BARREL_INNER: f64 = 165.0;
    const CORE_BARREL_OUTER: f64 = 170.0;
    const REFLECTOR_OUTER: f64 = 195.0;
    const VESSEL_OUTER: f64 = 220.0;
    const RADIUS_BATCH_1: f64 = 60.0;
    const RADIUS_BATCH_2: f64 = 120.0;
    const CONTROL_POSITIONS: &[(i32, i32)] = &[
        (0, 0),
        (0, 4),
        (0, -4),
        (4, 0),
        (-4, 0),
        (4, 4),
        (4, -4),
        (-4, 4),
        (-4, -4),
        (0, 7),
        (0, -7),
        (7, 0),
        (-7, 0),
        (3, 6),
        (3, -6),
        (-3, 6),
        (-3, -6),
        (6, 3),
        (6, -3),
        (-6, 3),
        (-6, -3),
    ];
    const MAT_FRESH: usize = 0;
    const MAT_MID: usize = 1;
    const MAT_BURNT: usize = 2;
    const MAT_CONTROL: usize = 3;
    const MAT_WATER: usize = 4;
    const MAT_STEEL: usize = 5;

    let materials = vec![
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
    let push = |s: &mut Vec<Surface>, surface: Surface| -> usize {
        let idx = s.len();
        s.push(surface);
        idx
    };
    let cyl = |cx, cy, r| Surface::CylinderZ {
        center_x: cx,
        center_y: cy,
        radius: r,
        bc: BoundaryCondition::Transmission,
    };
    let s_barrel_inner = push(&mut surfaces, cyl(0.0, 0.0, CORE_BARREL_INNER));
    let s_barrel_outer = push(&mut surfaces, cyl(0.0, 0.0, CORE_BARREL_OUTER));
    let s_reflector_outer = push(&mut surfaces, cyl(0.0, 0.0, REFLECTOR_OUTER));
    let _s_vessel_outer = push(
        &mut surfaces,
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: VESSEL_OUTER,
            bc: BoundaryCondition::Vacuum,
        },
    );

    let cp1_radius = 7.6 * ASSEMBLY_PITCH;
    for j in -8_i32..=8 {
        for i in -8_i32..=8 {
            let cx = (i as f64) * ASSEMBLY_PITCH;
            let cy = (j as f64) * ASSEMBLY_PITCH;
            let r = (cx * cx + cy * cy).sqrt();
            if r > cp1_radius - ASSEMBLY_PITCH * 0.5 {
                continue;
            }
            let mat = if CONTROL_POSITIONS.contains(&(i, j)) {
                MAT_CONTROL
            } else if r < RADIUS_BATCH_1 {
                MAT_FRESH
            } else if r < RADIUS_BATCH_2 {
                MAT_MID
            } else {
                MAT_BURNT
            };
            let s = push(&mut surfaces, cyl(cx, cy, ASSEMBLY_RADIUS));
            cells.push(
                Cell::new(
                    CellId(cells.len() as u32),
                    inside(s),
                    CellFill::Material(mat as u32),
                )
                .with_aabb_from_region(&surfaces),
            );
            cell_materials.push(mat);
        }
    }

    cells.push(
        Cell::new(
            CellId(cells.len() as u32),
            inside(s_barrel_inner),
            CellFill::Material(MAT_WATER as u32),
        )
        .with_aabb_from_region(&surfaces),
    );
    cell_materials.push(MAT_WATER);
    cells.push(
        Cell::new(
            CellId(cells.len() as u32),
            between(s_barrel_inner, s_barrel_outer),
            CellFill::Material(MAT_STEEL as u32),
        )
        .with_aabb_from_region(&surfaces),
    );
    cell_materials.push(MAT_STEEL);
    cells.push(
        Cell::new(
            CellId(cells.len() as u32),
            between(s_barrel_outer, s_reflector_outer),
            CellFill::Material(MAT_WATER as u32),
        )
        .with_aabb_from_region(&surfaces),
    );
    cell_materials.push(MAT_WATER);
    cells.push(
        Cell::new(
            CellId(cells.len() as u32),
            between(s_reflector_outer, _s_vessel_outer),
            CellFill::Material(MAT_STEEL as u32),
        )
        .with_aabb_from_region(&surfaces),
    );
    cell_materials.push(MAT_STEEL);

    let viewport = Viewport {
        x_min: -VESSEL_OUTER * 1.25,
        x_max: VESSEL_OUTER * 1.25,
        y_min: -VESSEL_OUTER * 1.25,
        y_max: VESSEL_OUTER * 1.25,
        z_slice: 0.0,
        width: 90,
        height: 45,
    };
    print_ascii(
        &cells,
        &surfaces,
        &materials,
        |i| cell_materials[i],
        &viewport,
    );
}
