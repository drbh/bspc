//! Abstract interfaces for BSPC specification
//!
//! This module defines all trait abstractions used in the BSPC ecosystem.
//! Traits are pure interfaces - no concrete implementations.

pub mod matrix;
pub mod element;
pub mod backend;

pub use matrix::SparseMatrix;
#[cfg(feature = "alloc")]
pub use matrix::MatrixOperations;
pub use element::MatrixElement;
pub use backend::{StorageBackend, ChunkProcessor, Chunkable};