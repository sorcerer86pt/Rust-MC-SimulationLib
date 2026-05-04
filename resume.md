# rust-mc-sim — running status

A short, time-honest log of what's in the library and what each
piece is actually good for. Add to the end as new work lands; don't
rewrite history.

## How to read the numbers

Every benchmark row is tagged with a scope:

- `[pin]`  = single PWR pin cell, 3 cells, full ENDF/B-VII.1 + S(α,β).
- `[asm]`  = 17 × 17 fuel assembly, 529 cells, same nuclide set.
- `[core]` = CP1 whole-core preview (geometry only, no transport).
- `[micro]` = isolated kernel / per-pixel / per-cell micro-benchmark.

A number quoted without scope is a bug. The repeated pattern of this
session: a kernel-level optimisation (BVH) is ~zero on `[pin]`, a
big win on `[asm]`. Quote the right scope.

## Latest

### BVH wired through cell-finding (commit `c60c4dd`)

Cell-finding used to be a linear scan over `cells: &[Cell]` even
though `geometry::bvh` had a working O(log N) tree. The tree only
existed; nothing called it from a hot loop. Wired it.

Done:

- `Aabb::intersection`, `Region::aabb(&surfaces)`, and
  `Cell::with_aabb_from_region(&surfaces)`. Cell AABBs now follow
  from the cell's region by walking the boolean tree against each
  surface's own AABB. Half-spaces of bounded surfaces shrink the
  AABB on bounded axes; outside-half-spaces and complements leave
  it infinite (no constraint). Intersections clip, unions enlarge.
  Tested on the cylinder + plane combos that show up in PWR pin
  geometries.
- `Bvh::build` keeps cells with infinite-axis AABBs (Z-cylinders),
  the splitter ignores infinite axes when picking a sort axis.
- `Bvh::find_cell` traverses both subtrees and returns the
  *lowest-index matching cell*. Same semantics as the linear scan
  in `ray::find_cell`, so OpenMC-style geometries — "water = inside
  barrel" with assemblies listed earlier in the cells vec to shadow
  it — render and transport identically with or without the BVH.
- `ray::find_cell_bvh` / `ray::find_cell_opt` / `ray::trace_step_opt`
  expose the BVH-aware path. Hot loops in `transport::simulate`,
  `transport::fixed_source`, `photon::transport` and
  `preview::render_top_down` build a BVH once and call the BVH
  path everywhere. No callsites left on the linear path inside hot
  loops.

### ASCII renderer + auto-colour-from-name (commit `c60c4dd`)

`preview::ascii::print_ascii` paints the geometry into a terminal
using ANSI 24-bit background colours derived from each material's
name (light water blue, fuel red, MOX orange, lead dark, RCCA
yellow, etc). Same keyword tree drives the windowed previewer's
`MaterialPalette::for_materials`. Useful as a fast diagnostic over
SSH or in CI; works without any GUI dep at runtime.

`auto_color_from_name(name)` and `ascii_glyph_for_name(name)` are
both `pub`, so callers can ship their own previewers / diagnostics
without re-implementing the keyword logic.

### 17 × 17 PWR assembly benchmark (commit `05b2d30`)

`examples/16_pwr_assembly_keff` builds the full Westinghouse 17×17
footprint — 264 fuel pins + 24 guide tubes + 1 instrumentation
thimble (latter 25 water-filled), reflective walls, all 9 ENDF/B-VII.1
nuclides + S(α,β) on H. Cell count ≈ 530, surface count ≈ 532.

Concrete head-to-head with the bare pin cell on the same
nuclide set:

| metric                  | `[pin]` (3 cells) | `[asm]` (529 cells) |
|-------------------------|-------------------|---------------------|
| `k_∞`                   | 1.37585 ± 0.00233 | **1.40183 ± 0.00483** |
| ns / history            | 223 773           | 1 582 422           |
| collisions / history    | ~30               | 30.1                |
| sim time                | 67 s (300 k hist) | 63 s (40 k hist)    |

Reading:

- `[asm]` runs at **7.1 × the per-history cost** of `[pin]` despite
  having **177 × more cells**. That ratio is the BVH speedup —
  without it the per-history cost would scale roughly with cell
  count, putting the assembly run in the multi-hour range.
- `[asm]` k_∞ sits **+2 600 pcm above** the pin cell (1.402 vs
  1.376). Physically correct direction: 25 of the 289 lattice
  positions are water-filled guide tubes, so the assembly is more
  moderator-rich than a tight pin cell. Lands inside the
  1.39 – 1.41 band typical for fresh CP1 / Westinghouse 17 × 17 at
  HZP.
- σ is wider (±483 pcm vs ±233 pcm) because the assembly run used
  20 active batches × 2000 particles vs the pin cell's 60 × 5000.
  σ ∝ 1/√N, expected √7.5 ≈ 2.7× wider, observed 2.07×. Fine.

Pin cell results were **bit-identical** before and after the BVH
wiring (k_∞ = 1.37585 ± 0.00233 in both cases) — the BVH preserves
first-match semantics so the eigenvalue power iteration sees an
identical RNG stream. The wiring is a pure speedup, not a physics
change.

## What this lets you do now that you couldn't before

- Run k_∞ on a real 17 × 17 PWR fuel assembly with full ENDF/B-VII.1
  physics in tens of seconds to tens of minutes, depending on batch
  size. Was effectively infeasible before the BVH.
- Render any CSG geometry to a terminal with colour-per-material,
  no window manager needed. ASCII output suitable for SSH / CI / pipe
  to a file.
- Render any CSG geometry to a window with drag-resize zoom (constant
  cm/pixel), scroll-wheel zoom, R reset, L legend popup. Closes
  cleanly on Esc / window-X.

## What's still on the wish list

- **Whole-core 157-assembly CP1 with pin-resolved detail** —
  ≈ 100 k cells. The BVH covers cell-finding, but the surface
  count grows with the pin count too (one cylinder per pin × 2 +
  walls = ~80 k surfaces); the per-call surface-evals loop in
  `find_cell` still scales O(surfaces). For pin-resolved whole-core
  the next move is either (a) restrict surface evaluation to surfaces
  referenced by AABB-overlapping cells, or (b) accept that
  assembly-homogenised cores are the real production geometry.
- **Coupled n-γ inside the neutron transport loop** — the photon
  HDF5 reader carries `PhotonProduct` data but the neutron collision
  dispatch doesn't bank γ's yet. Adding a `gamma_bank: &mut Vec<…>`
  field on the collision call site is the obvious next change.
- **IFBA / Optimized ZIRLO for RFA-2 fidelity** — bare 17 × 17 in
  example 16 uses Zircaloy-4 with Zr-90/91/92/94 only. Real RFA-2
  is Optimized ZIRLO (~1 wt % Nb, 1 wt % Sn, 0.1 wt % Fe; would
  need an Nb-93 nuclide); IFBA is ZrB₂-coated pellets at ≈ 80 — 116
  selected positions, depresses BOC k_∞ by 5 000 — 7 000 pcm. Both
  add only material-list entries and an extra cell type; no
  transport-loop change needed.
- **Time-dependent neutron transport** — both drivers are
  steady-state (k-eigenvalue, fixed-source). Power excursions and
  pulsed experiments need a time-domain driver.
