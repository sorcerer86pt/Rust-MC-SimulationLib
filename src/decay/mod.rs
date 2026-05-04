pub mod chain;

#[cfg(feature = "chain")]
pub mod openmc_xml;

pub use chain::{DecayChain, DecayMode, DecayNuclide, ReactionTarget};

#[cfg(feature = "chain")]
pub use openmc_xml::{ChainXmlError, load_chain_xml};
