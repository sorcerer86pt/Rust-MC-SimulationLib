//! OpenMC `chain.xml` parser. Implements the schema documented at
//! <https://docs.openmc.org/en/stable/io_formats/depletion_chain.html>.
//!
//! Output is a [`crate::decay::DecayChain`]. The parser handles:
//!   * `<nuclide name half_life decay_energy>`
//!   * `<decay type target branching_ratio Q>`
//!   * `<reaction type target Q branching_ratio>`
//!   * `<neutron_fission_yields>` with `<energies>` and one
//!     `<fission_yields energy="…">` per incident energy, each with
//!     `<products>` (whitespace-separated) and `<data>`
//!     (whitespace-separated yields parallel to `<products>`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

use crate::decay::chain::{
    DecayChain, DecayMode, DecayNuclide, ReactionChannel, ReactionTarget,
};
use crate::fission_yields::{FissionYields, YieldTable};

#[derive(Debug)]
pub enum ChainXmlError {
    Io(std::io::Error),
    Xml(quick_xml::Error),
    Parse(String),
}

impl std::fmt::Display for ChainXmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainXmlError::Io(e) => write!(f, "I/O error reading chain.xml: {e}"),
            ChainXmlError::Xml(e) => write!(f, "XML parse error in chain.xml: {e}"),
            ChainXmlError::Parse(msg) => write!(f, "chain.xml schema error: {msg}"),
        }
    }
}

impl std::error::Error for ChainXmlError {}

impl From<std::io::Error> for ChainXmlError {
    fn from(e: std::io::Error) -> Self {
        ChainXmlError::Io(e)
    }
}

impl From<quick_xml::Error> for ChainXmlError {
    fn from(e: quick_xml::Error) -> Self {
        ChainXmlError::Xml(e)
    }
}

/// Load an OpenMC depletion chain from `path`.
pub fn load_chain_xml(path: impl AsRef<Path>) -> Result<DecayChain, ChainXmlError> {
    let f = File::open(path)?;
    let buf = BufReader::new(f);
    parse_chain_xml(buf)
}

/// Parse an OpenMC depletion chain from any [`std::io::BufRead`].
pub fn parse_chain_xml<R: std::io::BufRead>(reader: R) -> Result<DecayChain, ChainXmlError> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut chain = DecayChain::new();

    // Streaming state.
    let mut current: Option<DecayNuclide> = None;
    let mut in_fy = false;
    let mut fy_energies: Vec<f64> = Vec::new();
    let mut fy_tables: BTreeMap<u64, YieldTable> = BTreeMap::new();
    let mut current_fy_energy: Option<f64> = None;
    let mut current_fy_products: Vec<String> = Vec::new();
    let mut current_fy_data: Vec<f64> = Vec::new();
    let mut text_target: Option<TextTarget> = None;

    loop {
        match xml.read_event_into(&mut buf)? {
            Event::Start(e) => {
                let name = e.name();
                match name.as_ref() {
                    b"nuclide" => {
                        current = Some(start_nuclide(&e)?);
                    }
                    b"decay" => {
                        if let Some(nuc) = current.as_mut() {
                            nuc.decay_modes.push(parse_decay(&e)?);
                        }
                    }
                    b"reaction" => {
                        if let Some(nuc) = current.as_mut() {
                            nuc.reactions.push(parse_reaction(&e)?);
                        }
                    }
                    b"neutron_fission_yields" => {
                        in_fy = true;
                        fy_energies.clear();
                        fy_tables.clear();
                    }
                    b"energies" if in_fy => {
                        text_target = Some(TextTarget::Energies);
                    }
                    b"fission_yields" if in_fy => {
                        let e_attr = require_attr(&e, "energy")?;
                        current_fy_energy = Some(parse_f64(&e_attr)?);
                        current_fy_products.clear();
                        current_fy_data.clear();
                    }
                    b"products" if in_fy => {
                        text_target = Some(TextTarget::Products);
                    }
                    b"data" if in_fy => {
                        text_target = Some(TextTarget::Data);
                    }
                    _ => {}
                }
            }
            Event::Empty(e) => {
                let name = e.name();
                match name.as_ref() {
                    b"decay" => {
                        if let Some(nuc) = current.as_mut() {
                            nuc.decay_modes.push(parse_decay(&e)?);
                        }
                    }
                    b"reaction" => {
                        if let Some(nuc) = current.as_mut() {
                            nuc.reactions.push(parse_reaction(&e)?);
                        }
                    }
                    _ => {}
                }
            }
            Event::Text(t) => {
                if let Some(target) = text_target {
                    let s = t.unescape()?;
                    match target {
                        TextTarget::Energies => {
                            for tok in s.split_whitespace() {
                                fy_energies.push(parse_f64(tok)?);
                            }
                        }
                        TextTarget::Products => {
                            current_fy_products
                                .extend(s.split_whitespace().map(str::to_string));
                        }
                        TextTarget::Data => {
                            for tok in s.split_whitespace() {
                                current_fy_data.push(parse_f64(tok)?);
                            }
                        }
                    }
                }
            }
            Event::End(e) => match e.name().as_ref() {
                b"nuclide" => {
                    if let Some(nuc) = current.take() {
                        chain.push(nuc);
                    }
                }
                b"fission_yields" if in_fy => {
                    if let Some(energy) = current_fy_energy.take() {
                        if current_fy_products.len() != current_fy_data.len() {
                            return Err(ChainXmlError::Parse(format!(
                                "fission_yields at {energy} eV: products {} ≠ data {}",
                                current_fy_products.len(),
                                current_fy_data.len()
                            )));
                        }
                        let table = YieldTable {
                            products: std::mem::take(&mut current_fy_products),
                            yields: std::mem::take(&mut current_fy_data),
                        };
                        fy_tables.insert(energy.to_bits(), table);
                    }
                }
                b"neutron_fission_yields" => {
                    in_fy = false;
                    if let Some(nuc) = current.as_mut() {
                        let mut fy = FissionYields::new();
                        for (k, v) in std::mem::take(&mut fy_tables) {
                            fy.tables.insert(k, v);
                        }
                        nuc.fission_yields = Some(fy);
                    }
                }
                b"energies" | b"products" | b"data" => {
                    text_target = None;
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(chain)
}

#[derive(Copy, Clone)]
enum TextTarget {
    Energies,
    Products,
    Data,
}

fn start_nuclide(e: &BytesStart<'_>) -> Result<DecayNuclide, ChainXmlError> {
    let name = require_attr(e, "name")?;
    let mut nuc = DecayNuclide::new(name);
    if let Some(hl) = optional_attr(e, "half_life")? {
        nuc.half_life = Some(parse_f64(&hl)?);
    }
    if let Some(en) = optional_attr(e, "decay_energy")? {
        nuc.decay_energy = parse_f64(&en)?;
    }
    Ok(nuc)
}

fn parse_decay(e: &BytesStart<'_>) -> Result<DecayMode, ChainXmlError> {
    let mode = require_attr(e, "type")?;
    let target = optional_attr(e, "target")?
        .map(ReactionTarget::Nuclide)
        .unwrap_or(ReactionTarget::Lost);
    let branching_ratio = optional_attr(e, "branching_ratio")?
        .map(|s| parse_f64(&s))
        .transpose()?
        .unwrap_or(1.0);
    Ok(DecayMode {
        mode,
        target,
        branching_ratio,
    })
}

fn parse_reaction(e: &BytesStart<'_>) -> Result<ReactionChannel, ChainXmlError> {
    let mt = require_attr(e, "type")?;
    let target = optional_attr(e, "target")?
        .map(ReactionTarget::Nuclide)
        .unwrap_or(ReactionTarget::Lost);
    let q_value = optional_attr(e, "Q")?
        .map(|s| parse_f64(&s))
        .transpose()?
        .unwrap_or(0.0);
    let branching_ratio = optional_attr(e, "branching_ratio")?
        .map(|s| parse_f64(&s))
        .transpose()?
        .unwrap_or(1.0);
    Ok(ReactionChannel {
        mt,
        target,
        q_value,
        branching_ratio,
    })
}

fn require_attr(e: &BytesStart<'_>, key: &str) -> Result<String, ChainXmlError> {
    optional_attr(e, key)?.ok_or_else(|| {
        let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
        ChainXmlError::Parse(format!("<{tag}>: missing required attribute '{key}'"))
    })
}

fn optional_attr(e: &BytesStart<'_>, key: &str) -> Result<Option<String>, ChainXmlError> {
    for attr in e.attributes().with_checks(false) {
        let attr = attr.map_err(|err| ChainXmlError::Parse(err.to_string()))?;
        if attr.key.as_ref() == key.as_bytes() {
            let v = attr.unescape_value().map_err(ChainXmlError::Xml)?;
            return Ok(Some(v.into_owned()));
        }
    }
    Ok(None)
}

fn parse_f64(s: &str) -> Result<f64, ChainXmlError> {
    s.trim()
        .parse::<f64>()
        .map_err(|e| ChainXmlError::Parse(format!("not a real number: '{s}' ({e})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<depletion_chain>
  <nuclide name="U235" half_life="2.221e16" decay_energy="4.572e6">
    <decay type="alpha" target="Th231" branching_ratio="1.0"/>
    <reaction type="(n,gamma)" target="U236" Q="6545200.0" branching_ratio="1.0"/>
    <reaction type="fission" Q="1.93e8" branching_ratio="1.0">
      <neutron_fission_yields>
        <energies>0.0253 500000.0</energies>
        <fission_yields energy="0.0253">
          <products>Cs137 Sr90</products>
          <data>0.062 0.058</data>
        </fission_yields>
        <fission_yields energy="500000.0">
          <products>Cs137 Sr90</products>
          <data>0.060 0.055</data>
        </fission_yields>
      </neutron_fission_yields>
    </reaction>
  </nuclide>
  <nuclide name="Cs137" half_life="9.488e8">
    <decay type="beta-" target="Ba137_m1" branching_ratio="0.946"/>
    <decay type="beta-" target="Ba137" branching_ratio="0.054"/>
  </nuclide>
  <nuclide name="Th231"/>
  <nuclide name="Ba137"/>
  <nuclide name="Ba137_m1"/>
  <nuclide name="U236"/>
  <nuclide name="Sr90"/>
</depletion_chain>
"#;

    #[test]
    fn parses_minimal_chain() {
        let chain = parse_chain_xml(SAMPLE.as_bytes()).expect("parse");
        let u235_idx = chain.index_of("U235").expect("U235 in chain");
        let u235 = &chain.nuclides[u235_idx];
        assert!((u235.half_life.unwrap() - 2.221e16).abs() < 1e10);
        assert_eq!(u235.decay_modes.len(), 1);
        assert_eq!(u235.decay_modes[0].mode, "alpha");
        assert_eq!(
            u235.decay_modes[0].target,
            ReactionTarget::Nuclide("Th231".into())
        );
        assert_eq!(u235.reactions.len(), 2);
        let fission = u235
            .reactions
            .iter()
            .find(|r| r.mt == "fission")
            .expect("fission rxn");
        // Cross-check fission target is Lost (no `target=` on a
        // fission reaction; products come from the yield table).
        assert_eq!(fission.target, ReactionTarget::Lost);

        // Yields parsed.
        let fy = u235.fission_yields.as_ref().expect("fission yields");
        let products_thermal = fy.products_at_energy(0.0253);
        let cs137 = products_thermal
            .iter()
            .find(|(p, _)| p == "Cs137")
            .expect("Cs137 yield")
            .1;
        assert!((cs137 - 0.062).abs() < 1e-12);
    }

    #[test]
    fn cs137_branching_ratios_sum_to_one() {
        let chain = parse_chain_xml(SAMPLE.as_bytes()).expect("parse");
        let cs = chain.index_of("Cs137").expect("Cs137 in chain");
        let cs = &chain.nuclides[cs];
        let total: f64 = cs.decay_modes.iter().map(|m| m.branching_ratio).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn transmutation_matrix_routes_fission_yields() {
        let chain = parse_chain_xml(SAMPLE.as_bytes()).expect("parse");
        // φ = 1e14 n/cm²s, σ_f(U235) = 585 b at thermal.
        let m = chain.build_transmutation_matrix(1.0e14, 0.0253, |idx, mt| {
            if chain.nuclides[idx].name == "U235" {
                match mt {
                    "fission" => 585.0,
                    "(n,gamma)" => 99.0,
                    _ => 0.0,
                }
            } else {
                0.0
            }
        });
        let cs_idx = chain.index_of("Cs137").unwrap();
        let u_idx = chain.index_of("U235").unwrap();
        let entry = m[(cs_idx, u_idx)];
        // Expect ~ 585 b · 1e-24 · 1e14 · 0.062 = 3.627e-9 /s
        assert!(entry > 0.0, "no fission→Cs137 routing");
        let want = 585.0 * 1.0e-24 * 1.0e14 * 0.062;
        assert!(
            (entry / want - 1.0).abs() < 1e-3,
            "Cs137 yield routing wrong: got {entry}, want {want}"
        );
    }
}
