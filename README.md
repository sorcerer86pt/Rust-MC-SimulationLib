# rust-mc-sim

Reusable building blocks for Monte Carlo simulation in Rust.

Truncated SVD with cache-friendly reconstruction, CP/PARAFAC for
3-tensors, Ducru-weighted off-grid temperature interpolation, and
log-decimated CDF inverse-transform sampling. An optional
`nuclear` feature ships the OpenMC-compatible HDF5 reader,
windowed-multipole evaluator (Humlicek W4 Faddeeva), and S(α,β)
thermal scattering kernels. An optional `parallel` feature exposes
rayon-driven batch APIs for at-scale loads (200k-nuclide
depletion-style libraries, large CP-decomposition sweeps).

This is the **math + data foundation** crate. The full transport
engine (geometry, materials, k-eigenvalue solver, photon transport,
GPU kernels) lives in
[**open_rust_mc**](https://github.com/sorcerer86pt/open_rust_mc),
which depends on this crate for its compression algorithms.

Extracted and generalised from
[**open_rust_mc**](https://github.com/sorcerer86pt/open_rust_mc), where
the same algorithms power a continuous-energy Monte Carlo neutron
transport engine. Every algorithm here was validated against OpenMC
0.15.3 in the parent project (Godiva HMF-001 within ICSBEP σ_exp,
PWR pin cell within 51 pcm cross-code; full numbers in the parent
project's paper).

## What's in the box

### Pure math (no extra deps beyond [`faer`])

| Module | What it does |
|---|---|
| [`rust_mc_sim::svd`] | Truncated SVD via faer + a cache-friendly `SvdKernel` with optional log-uniform hash index |
| [`rust_mc_sim::table`] | Production-baseline pointwise table with log-log interpolation; OpenMC-style stochastic temperature pseudo-interpolation. The right pick when you want byte-exact values at grid points |
| [`rust_mc_sim::cp`] | CP/PARAFAC decomposition of a 3-tensor (greedy rank-1 deflation) |
| [`rust_mc_sim::ducru`] | Ducru-2017 free-Doppler reconstruction weights, raw and partition-of-unity |
| [`rust_mc_sim::cdf`] | Log-decimated CDF for inverse-transform sampling of categorical outcomes that depend on a continuous coordinate |
| [`rust_mc_sim::rng`] | PCG-64 RNG used by `cdf` sampling and the nuclear layer |
| [`rust_mc_sim::batch`] | Sequential and rayon-parallel batch APIs for at-scale loads (200k-nuclide depletion-style libraries) |

### Nuclear-data layer (`feature = "nuclear"`, adds [`hdf5-pure`])

| Module | What it does |
|---|---|
| [`rust_mc_sim::nuclear::wmp`] | Windowed Multipole evaluator with Humlicek W4 Faddeeva and HDF5 loader |
| [`rust_mc_sim::nuclear::thermal`] | S(α,β) thermal scattering data structures and sampling |
| [`rust_mc_sim::nuclear::hdf5`] | Pure-Rust OpenMC HDF5 reader: cross sections, level metadata, angular distributions, energy distributions, URR, S(α,β) |

## Quick start

```toml
# Cargo.toml
[dependencies]
rust-mc-sim = "0.1"
# or, with the nuclear-data layer:
# rust-mc-sim = { version = "0.1", features = ["nuclear"] }
```

### Pointwise table — the production baseline

```rust
use rust_mc_sim::PointwiseTable;

let xs = vec![1.0, 10.0, 100.0, 1000.0];
let vs = vec![2.0, 4.0, 8.0, 16.0];
let mut t = PointwiseTable::new(xs, vs);
t.build_hash(8192);              // O(1) lookup once built
assert!((t.lookup(50.0) - 6.5).abs() < 0.5);
```

For OpenMC-style stochastic temperature pseudo-interpolation:

```rust,ignore
use rust_mc_sim::{PointwiseTable, StochTempTable};

let lo = PointwiseTable::new(/* T=600 K */);
let hi = PointwiseTable::new(/* T=900 K */);
let stoch = StochTempTable::stochastic(lo, hi, 750.0, 600.0, 900.0);

// Single-channel lookup with internal pick:
let v = stoch.lookup(some_x);

// Multi-channel: draw one pick per physics event, share across channels:
let pick = stoch_elastic.draw_pick();
let el = stoch_elastic.lookup_at_idx_with_pick(x, idx, pick);
let cap = stoch_capture.lookup_at_idx_with_pick(x, idx, pick);
let fis = stoch_fission.lookup_at_idx_with_pick(x, idx, pick);
```

### Truncated SVD with off-column reconstruction

```rust,ignore
use std::sync::Arc;
use rust_mc_sim::SvdKernel;

let n_rows = 4000;
let n_cols = 6;                                                   // 6 training columns
let row_axis: Arc<[f64]> = Arc::from(/* 4000 sorted-asc f64 */);

let kernel = SvdKernel::from_data(&data, row_axis, n_rows, n_cols, 5);

// At a training column:
let coeffs = kernel.coeffs_at_col(2);
let value = kernel.reconstruct_at(kernel.row_index(some_x), &coeffs);

// Off-grid target column (Ducru-weighted):
let column_values = vec![300.0, 600.0, 900.0, 1200.0, 1500.0, 2500.0];
let coeffs = kernel.ducru_coeffs(&column_values, /* target = */ 750.0);
let value = kernel.reconstruct_at(kernel.row_index(some_x), &coeffs);
```

### CP/PARAFAC of a 3-tensor

```rust,ignore
use rust_mc_sim::{cp_greedy_rank1, relative_l2_error};

// tensor[i * n_b * n_c + t * n_c + l], row-major
let cp = cp_greedy_rank1(&tensor, n_a, n_b, n_c, /* max_rank = */ 8, 200, 1e-9);

let recon5 = cp.reconstruct(5);
let l2 = relative_l2_error(&tensor, &recon5);
println!("rank-5 relative L2 = {l2:.2e}");
```

### Log-decimated CDF + inverse-transform sampling

```rust
use rust_mc_sim::cdf::LogDecimatedCdf;

let xs: Vec<f64> = (0..50).map(|i| 1.0 * 1.1f64.powi(i)).collect();
let n_cat = 3;
let mut intensities = vec![vec![0.0_f64; xs.len()]; n_cat];
for (j, &x) in xs.iter().enumerate() {
    intensities[0][j] = 1.0 / x;
    intensities[1][j] = (x.ln() / 5.0).max(0.0);
    intensities[2][j] = 1.0;
}
let cdf = LogDecimatedCdf::from_intensities(&intensities, &xs, 200);
let k = cdf.sample(10.0, 0.42);
assert!(k < 3);
```

### Off-grid Ducru blending (raw vs partition-of-unity)

```rust,ignore
use rust_mc_sim::{ducru_unity_weights, nearest_k_columns};

let library_temps = vec![294.0, 600.0, 900.0, 1200.0, 1500.0, 2500.0];
let target = 750.0;

// 3-point unity-normalised weights on the nearest three columns
let chosen = nearest_k_columns(&library_temps, target, 3);
let sub: Vec<f64> = chosen.iter().map(|&i| library_temps[i]).collect();
let weights = ducru_unity_weights(&sub, target);

// Σ weights = 1, exact at training columns
let s: f64 = weights.iter().sum();
assert!((s - 1.0).abs() < 1e-12);
```

## Scaling at 200k+

For depletion-style workloads with hundreds of thousands of
independent decompositions (full-actinide libraries, all-MT sweeps,
multi-temperature batch builds), enable the `parallel` feature and
use the [`batch`] APIs:

```toml
rust-mc-sim = { version = "0.1", features = ["parallel"] }
```

```rust,ignore
use rust_mc_sim::batch::{from_data_many_par, KernelInput};

let inputs: Vec<KernelInput<'_>> = nuclides
    .iter()
    .map(|n| KernelInput { /* … */ })
    .collect();
let kernels = from_data_many_par(&inputs);   // fans out across rayon threads
```

Measured on a 20-core Intel mobile workstation, 80 rows × 6 columns,
rank 5 (after warm-up):

| N kernels | sequential | parallel | speedup | memory |
|---:|---:|---:|---:|---:|
| 10 000  | 2.9 s   | 0.075 s | **38.5×** | 33 MB |
| projected to 200 000 | 58 s | 1.5 s | — | 0.6 GB |

A few things matter at this scale:

* **Hash index is opt-in.** `SvdKernel::from_data` builds a
  [`LogHashIndex`] (32 KB / kernel at default 8192 bins) when the row
  axis has more than 100 points. At 200k kernels this is ~6.4 GB of
  hash budget. `SvdKernel::from_factors` builds **no** hash; call
  [`SvdKernel::build_hash`] selectively for the few percent of
  kernels in the hot path of your transport loop.
* **Memory accounting.** Use [`batch::total_kernel_bytes`],
  [`batch::total_cp_bytes`], [`batch::total_cdf_bytes`] for
  load-time sanity checks: did I allocate the expected amount, or did
  something explode?
* **HDF5 I/O is your bottleneck before the SVD is.** The example
  above measures only the decomposition cost; reading 200k HDF5 files
  sequentially is minutes-scale. If you control the layout, batch
  multiple nuclides into a single HDF5 (the OpenMC convention) so
  one open + many group reads dominates.
* **Persist factor pairs to disk.** [`SvdKernel::from_factors`] takes
  pre-computed `(basis, vt)` so you can run the SVD pass once per
  library version and reload kernels in O(N rows) per kernel
  afterwards. Cuts the load-time cost from "minutes" to "seconds".

A worked example at 10k–200k scale lives in
[`examples/05_stress_200k.rs`](examples/05_stress_200k.rs); run it with:

```bash
cargo run --release --features parallel --example 05_stress_200k
RUST_MC_SIM_STRESS_N=50000 cargo run --release --features parallel \
    --example 05_stress_200k
```

## Status and audience

This crate is the algorithm/data-structure layer extracted from
`open_rust_mc`, not a public-API roadmap or a tutorial. The
algorithms have been validated end-to-end inside the parent
project; the public surface here mirrors the original module
names and signatures wherever possible, so anyone reading
`open_rust_mc` source can map calls 1-to-1.

What this is good for, in approximate priority order:

1. Decomposing tabulated multi-T scientific data (cross sections,
   spectra, opacities, flow fields…) into a compact basis +
   coefficient form.
2. Off-library reconstruction at arbitrary target temperatures
   via Ducru weights, with sub-percent error on smooth data and
   peak-preserving behaviour on resonance-dominated channels.
3. Sampling categorical outcomes (level selection, channel
   selection…) with probabilities that depend on a continuous
   coordinate, via the log-decimated CDF.
4. Reading OpenMC HDF5 nuclear-data libraries from pure Rust
   (no C dependency) via the `nuclear` feature.

## License

MIT. See [`LICENSE`](LICENSE).

## Provenance

Extracted from
[**sorcerer86pt/open_rust_mc**](https://github.com/sorcerer86pt/open_rust_mc).
The full validation story (Godiva HMF-001 ICSBEP benchmark, PWR pin
cell vs OpenMC 0.15.3, on- vs off-library sweeps, GPU CDF/synth
results) lives in the parent project's `paper/main.tex`.
