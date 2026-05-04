//! Error type for the I/O-bearing `nuclear` modules.

use std::fmt;

/// Errors emitted by the optional nuclear-data modules.
#[derive(Debug)]
pub enum NuclearError {
    /// HDF5 I/O failure with a contextual path + diagnostic.
    Hdf5 { path: String, detail: String },
    /// Dimension mismatch between expected and actual shape.
    DimensionMismatch { expected: String, got: String },
    /// Underlying I/O error.
    Io(std::io::Error),
}

impl fmt::Display for NuclearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NuclearError::Hdf5 { path, detail } => {
                write!(f, "HDF5 error reading {path}: {detail}")
            }
            NuclearError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            NuclearError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for NuclearError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NuclearError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NuclearError {
    fn from(e: std::io::Error) -> Self {
        NuclearError::Io(e)
    }
}

/// Convenience alias.
pub type NuclearResult<T> = std::result::Result<T, NuclearError>;
