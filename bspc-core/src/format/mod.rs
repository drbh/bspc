//! Binary format definitions for BSPC file specification
//!
//! This module contains pure data structure definitions for the BSPC wire format.
//! No I/O operations or concrete implementations - only format specifications.

pub mod constants;
pub mod header;
pub mod metadata;

// Re-export format definitions
pub use header::{BspcHeader, DataType, MatrixFormat};
pub use metadata::{BspcMetadataHeader, LabelArrayHeader};