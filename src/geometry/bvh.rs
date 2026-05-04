//! Bounding Volume Hierarchy — O(log n) cell lookup.
//!
//! Builds a binary tree of AABBs over cells. Traversal skips entire
//! subtrees when the ray doesn't intersect the bounding box.
//! Construction uses the Surface Area Heuristic (SAH) for optimal splits.

use super::{Aabb, Cell, Surface, Vec3};

/// BVH node — either a leaf (single cell) or an internal node (two children).
#[derive(Debug)]
enum BvhNode {
    Leaf {
        cell_idx: usize,
        aabb: Aabb,
    },
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

/// The BVH acceleration structure.
pub struct Bvh {
    root: Option<BvhNode>,
}

impl Bvh {
    /// Build a BVH from a set of cells.
    pub fn build(cells: &[Cell]) -> Self {
        if cells.is_empty() {
            return Self { root: None };
        }

        // We accept cells with infinite-axis AABBs (e.g. CylinderZ
        // is infinite in z) — the splitter picks the largest *finite*
        // axis so infinite axes don't poison the partition.
        let mut entries: Vec<(usize, Aabb, Vec3)> = cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let aabb = c.aabb;
                // Centroid: substitute 0 on infinite axes so sort
                // ordering is well-defined.
                let cx = if aabb.center().x.is_finite() { aabb.center().x } else { 0.0 };
                let cy = if aabb.center().y.is_finite() { aabb.center().y } else { 0.0 };
                let cz = if aabb.center().z.is_finite() { aabb.center().z } else { 0.0 };
                (i, aabb, Vec3::new(cx, cy, cz))
            })
            .collect();

        if entries.is_empty() {
            return Self { root: None };
        }

        let root = build_recursive(&mut entries);
        Self { root: Some(root) }
    }

    /// Find which cell contains a point, using BVH acceleration.
    ///
    /// Semantics match the linear scan in [`crate::geometry::ray::find_cell`]:
    /// when several cells' regions all match `pos`, the one with the
    /// *lowest* index in `cells` wins. This is needed so OpenMC-style
    /// lazy geometries — e.g. "water = inside the core barrel" with
    /// assembly cells listed earlier in the cells vec to shadow it —
    /// render and transport identically with or without the BVH.
    pub fn find_cell(&self, pos: Vec3, surfaces: &[Surface], cells: &[Cell]) -> Option<usize> {
        let root = self.root.as_ref()?;
        let evals: Vec<f64> = surfaces.iter().map(|s| s.evaluate(pos)).collect();
        let mut best: Option<usize> = None;
        find_cell_recursive(root, pos, &evals, cells, &mut best);
        best
    }
}

fn find_cell_recursive(
    node: &BvhNode,
    pos: Vec3,
    surface_evals: &[f64],
    cells: &[Cell],
    best: &mut Option<usize>,
) {
    match node {
        BvhNode::Leaf { cell_idx, aabb } => {
            // Skip if a smaller index has already matched — the
            // answer can't change.
            if let Some(b) = *best
                && *cell_idx >= b
            {
                return;
            }
            if aabb.contains(pos) && cells[*cell_idx].contains(surface_evals) {
                *best = Some(match *best {
                    Some(b) => b.min(*cell_idx),
                    None => *cell_idx,
                });
            }
        }
        BvhNode::Internal { aabb, left, right } => {
            if !aabb.contains(pos) {
                return;
            }
            find_cell_recursive(left, pos, surface_evals, cells, best);
            find_cell_recursive(right, pos, surface_evals, cells, best);
        }
    }
}

/// Recursively build the BVH using midpoint splitting.
fn build_recursive(entries: &mut [(usize, Aabb, Vec3)]) -> BvhNode {
    if entries.len() == 1 {
        return BvhNode::Leaf {
            cell_idx: entries[0].0,
            aabb: entries[0].1,
        };
    }

    // Compute overall AABB
    let overall_aabb = entries
        .iter()
        .map(|(_, aabb, _)| *aabb)
        .reduce(Aabb::union)
        .expect("non-empty");

    if entries.len() == 2 {
        return BvhNode::Internal {
            aabb: overall_aabb,
            left: Box::new(BvhNode::Leaf {
                cell_idx: entries[0].0,
                aabb: entries[0].1,
            }),
            right: Box::new(BvhNode::Leaf {
                cell_idx: entries[1].0,
                aabb: entries[1].1,
            }),
        };
    }

    // Largest *finite* axis — infinite axes (e.g. cylinder z) shouldn't
    // dictate the split because every entry shares the same infinite
    // extent and sorting collapses.
    let extent = overall_aabb.max - overall_aabb.min;
    let safe = |v: f64| if v.is_finite() { v } else { f64::NEG_INFINITY };
    let ex = safe(extent.x);
    let ey = safe(extent.y);
    let ez = safe(extent.z);
    let split_axis = if ex >= ey && ex >= ez {
        0
    } else if ey >= ez {
        1
    } else {
        2
    };

    // Sort by centroid along the split axis
    entries.sort_by(|a, b| {
        let ca = match split_axis {
            0 => a.2.x,
            1 => a.2.y,
            _ => a.2.z,
        };
        let cb = match split_axis {
            0 => b.2.x,
            1 => b.2.y,
            _ => b.2.z,
        };
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Split at the midpoint
    let mid = entries.len() / 2;
    let (left_entries, right_entries) = entries.split_at_mut(mid);

    let left = build_recursive(left_entries);
    let right = build_recursive(right_entries);

    BvhNode::Internal {
        aabb: overall_aabb,
        left: Box::new(left),
        right: Box::new(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, Cell, CellFill, CellId};
    use crate::geometry::surface::BoundaryCondition;

    #[test]
    fn bvh_finds_cells_with_infinite_z_aabb() {
        // Two CylinderZ pin cells side by side. AABBs are infinite
        // on z; the BVH must still locate the right cell at z = 0.
        let surfaces = vec![
            Surface::CylinderZ {
                center_x: -1.0,
                center_y: 0.0,
                radius: 0.4,
                bc: BoundaryCondition::Transmission,
            },
            Surface::CylinderZ {
                center_x: 1.0,
                center_y: 0.0,
                radius: 0.4,
                bc: BoundaryCondition::Transmission,
            },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0))
                .with_aabb_from_region(&surfaces),
            Cell::new(CellId(1), cell::inside(1), CellFill::Material(1))
                .with_aabb_from_region(&surfaces),
        ];
        let bvh = Bvh::build(&cells);

        // Point inside left pin at z=0.
        let pos_left = Vec3::new(-1.0, 0.0, 0.0);
        assert_eq!(bvh.find_cell(pos_left, &surfaces, &cells), Some(0));
        // Point inside right pin at z=5 (well outside finite z range).
        let pos_right = Vec3::new(1.0, 0.0, 5.0);
        assert_eq!(bvh.find_cell(pos_right, &surfaces, &cells), Some(1));
        // Point in the gap at the origin — no cell.
        assert!(bvh.find_cell(Vec3::new(0.0, 0.0, 0.0), &surfaces, &cells).is_none());
    }
}
