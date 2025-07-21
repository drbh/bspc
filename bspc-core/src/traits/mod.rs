//! Abstract interfaces for BSPC specification
//!
//! This module defines all trait abstractions used in the BSPC ecosystem.
//! Traits are pure interfaces - no concrete implementations.

pub mod backend;
pub mod element;
pub mod matrix;

pub use backend::{ChunkProcessor, Chunkable, StorageBackend};
pub use element::MatrixElement;
#[cfg(feature = "alloc")]
pub use matrix::MatrixOperations;
pub use matrix::SparseMatrix;
