//! Memory-mapped file backend for .bspc files
//!
//! This module implements a memory-mapped storage backend that can read and write
//! sparse matrices to/from .bspc files using memory mapping for efficient access.
//!
//! # Architecture
//!
//! The module is split into three main components:
//! - `mmap_core`: Core memory mapping types and traits
//! - `matrix_operations`: Matrix operations, views, and iterators
//! - `file_io`: File I/O operations and streaming writers

// Declare submodules
pub(crate) mod file_io;
pub(crate) mod matrix_operations;
pub(crate) mod mmap_core;

// Re-export main public types
pub use file_io::BspcFile;
pub use matrix_operations::{DynamicMatrix, DynamicMatrixRowIterator, SubmatrixView};
pub use mmap_core::{MatrixElement, MmapMatrix};
