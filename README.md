# tensor-compress

Low-rank approximations of multi-way tabulated data, in pure Rust.

Truncated SVD with cache-friendly reconstruction, CP/PARAFAC for
3-tensors, Ducru-weighted off-grid temperature interpolation, and
log-decimated CDF inverse-transform sampling. An optional
`nuclear` feature ships the OpenMC-compatible HDF5 reader,
windowed-multipole evaluator (Humlicek W4 Faddeeva), and S(α,β)
thermal scattering kernels.

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
| [`tensor_compress::svd`] | Truncated SVD via faer + a cache-friendly `SvdKernel` with optional log-uniform hash index |
| [`tensor_compress::cp`] | CP/PARAFAC decomposition of a 3-tensor (greedy rank-1 deflation) |
| [`tensor_compress::ducru`] | Ducru-2017 free-Doppler reconstruction weights, raw and partition-of-unity |
| [`tensor_compress::cdf`] | Log-decimated CDF for inverse-transform sampling of categorical outcomes that depend on a continuous coordinate |
| [`tensor_compress::rng`] | PCG-64 RNG used by `cdf` sampling and the nuclear layer |

### Nuclear-data layer (`feature = "nuclear"`, adds [`hdf5-pure`])

| Module | What it does |
|---|---|
| [`tensor_compress::nuclear::wmp`] | Windowed Multipole evaluator with Humlicek W4 Faddeeva and HDF5 loader |
| [`tensor_compress::nuclear::thermal`] | S(α,β) thermal scattering data structures and sampling |
| [`tensor_compress::nuclear::hdf5`] | Pure-Rust OpenMC HDF5 reader: cross sections, level metadata, angular distributions, energy distributions, URR, S(α,β) |

## Quick start

```toml
# Cargo.toml
[dependencies]
tensor-compress = "0.1"
# or, with the nuclear-data layer:
# tensor-compress = { version = "0.1", features = ["nuclear"] }
```

### Truncated SVD with off-column reconstruction

```rust,ignore
use std::sync::Arc;
use tensor_compress::SvdKernel;

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
use tensor_compress::{cp_greedy_rank1, relative_l2_error};

// tensor[i * n_b * n_c + t * n_c + l], row-major
let cp = cp_greedy_rank1(&tensor, n_a, n_b, n_c, /* max_rank = */ 8, 200, 1e-9);

let recon5 = cp.reconstruct(5);
let l2 = relative_l2_error(&tensor, &recon5);
println!("rank-5 relative L2 = {l2:.2e}");
```

### Log-decimated CDF + inverse-transform sampling

```rust
use tensor_compress::cdf::LogDecimatedCdf;

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
use tensor_compress::{ducru_unity_weights, nearest_k_columns};

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

Dual-licensed under MIT or Apache-2.0, at your option. See
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).

## Provenance

Extracted from
[**sorcerer86pt/open_rust_mc**](https://github.com/sorcerer86pt/open_rust_mc).
The full validation story (Godiva HMF-001 ICSBEP benchmark, PWR pin
cell vs OpenMC 0.15.3, on- vs off-library sweeps, GPU CDF/synth
results) lives in the parent project's `paper/main.tex`.
