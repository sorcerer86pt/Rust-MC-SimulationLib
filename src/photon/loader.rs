//! HDF5 loader for OpenMC photon-per-element files (`photon/<Sym>.h5`).
//! Reads only the slim v1 set of channels — five XS arrays on a shared
//! energy grid. Form factors, Compton profiles, subshells, and the
//! Seltzer-Berger bremsstrahlung table are *not* read by this loader;
//! when bound-electron Compton or atomic relaxation lands in this
//! crate, this loader gets a sibling that pulls them in.

use std::path::Path;

use crate::error::{NuclearError, NuclearResult};
use crate::photon::data::PhotonElement;

/// Load one element's photon-interaction data from an OpenMC HDF5
/// file. The reader expects the file at `path` to contain a single
/// element group (e.g. `/H`, `/Pb`) at the root.
pub fn load_photon_element(path: &Path) -> NuclearResult<PhotonElement> {
    let mk_err = |detail: String| NuclearError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };
    let file = hdf5_pure::File::open(path).map_err(|e| mk_err(format!("open: {e}")))?;
    let root = file.root();

    let symbol = root
        .groups()
        .map_err(|e| mk_err(format!("cannot list root groups: {e}")))?
        .into_iter()
        .next()
        .ok_or_else(|| mk_err("no element group at root".into()))?;

    let element = root
        .group(&symbol)
        .map_err(|e| mk_err(format!("cannot open /{symbol}: {e}")))?;

    let z = match element
        .attrs()
        .map_err(|e| mk_err(format!("cannot read /{symbol} attrs: {e}")))?
        .get("Z")
    {
        Some(hdf5_pure::AttrValue::I64(z)) => *z as u32,
        Some(_) | None => return Err(mk_err(format!("/{symbol} missing Z attribute"))),
    };

    let energy = element
        .dataset("energy")
        .map_err(|e| mk_err(format!("cannot open /{symbol}/energy: {e}")))?
        .read_f64()
        .map_err(|e| mk_err(format!("cannot read /{symbol}/energy: {e}")))?;
    let n_e = energy.len();

    let coherent_xs = read_xs(&element, "coherent", n_e, &mk_err)?;
    let incoherent_xs = read_xs(&element, "incoherent", n_e, &mk_err)?;
    let photoelectric_xs = read_xs(&element, "photoelectric", n_e, &mk_err)?;
    let pair_production_nuclear_xs = read_xs(&element, "pair_production_nuclear", n_e, &mk_err)?;
    let pair_production_electron_xs = read_xs(&element, "pair_production_electron", n_e, &mk_err)?;

    Ok(PhotonElement {
        z,
        symbol,
        energy,
        coherent_xs,
        incoherent_xs,
        photoelectric_xs,
        pair_production_nuclear_xs,
        pair_production_electron_xs,
    })
}

fn read_xs<F>(
    element: &hdf5_pure::Group,
    channel: &str,
    n_e: usize,
    mk_err: &F,
) -> NuclearResult<Vec<f64>>
where
    F: Fn(String) -> NuclearError,
{
    // Some elements (low-Z) tabulate zero pair production. The
    // group may still exist with an `xs` dataset of length n_e
    // filled with zeros; if the group is genuinely absent, fill
    // zeros — this lets the loader handle Z=1 (no pair) without
    // crashing.
    let group = match element.group(channel) {
        Ok(g) => g,
        Err(_) => return Ok(vec![0.0; n_e]),
    };
    let ds = group
        .dataset("xs")
        .map_err(|e| mk_err(format!("cannot open {channel}/xs: {e}")))?;
    let xs = ds
        .read_f64()
        .map_err(|e| mk_err(format!("cannot read {channel}/xs: {e}")))?;
    if xs.len() != n_e {
        // Tail-align if the channel ships fewer points than the
        // master grid (OpenMC convention for threshold reactions).
        let mut padded = vec![0.0; n_e];
        let offset = n_e - xs.len();
        padded[offset..].copy_from_slice(&xs);
        Ok(padded)
    } else {
        Ok(xs)
    }
}
