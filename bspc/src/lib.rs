//! BSPC - High-performance Binary Sparse Matrix Implementation
//!
//! This library provides efficient sparse matrix storage and access using the BSPC format
//! with memory mapping, HTTP backends, and bloom filter optimizations.
//!
//! ## Architecture
//!
//! BSPC follows a clean specification/implementation separation:
//!
//! - **bspc-core**: Pure format specifications, traits, and validation (no I/O)
//! - **bspc**: Concrete implementations with I/O, networking, and optimizations
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use bspc::{BspcFile, ChunkConfig, MatrixElement};
//! 
//! fn example() -> Result<(), binsparse_rs::Error> {
//!     // Load a matrix with bloom filter optimization
//!     let config = ChunkConfig::default().with_bloom_hash_count(3);
//!     let matrix = BspcFile::read_matrix_with_bloom_filter("matrix.bspc", config)?;
//! 
//!     // Access elements efficiently
//!     let dimensions = matrix.dimensions();
//!     if let Some(value) = matrix.get_element(100, 200) {
//!         println!("matrix[100, 200] = {}", value.to_f64());
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **Memory-mapped I/O**: Zero-copy access to large sparse matrices
//! - **Bloom filters**: Skip empty chunks for faster sparse access
//! - **HTTP backend**: Stream matrices over HTTP with range requests
//! - **Metadata support**: Row/column labels with O(1) lookup
//! - **Type safety**: Strong typing with bspc-core abstractions

// Re-export core abstractions and format definitions  
pub use bspc_core::{
    // Core traits
    SparseMatrix, MatrixOperations, MatrixElement,
    // Format definitions  
    BspcHeader, DataType, MatrixFormat,
    // Error handling
    BspcError, Result, ErrorCategory,
    // Validation utilities
    validate_array_bounds, parse_range,
};

// binsparse_rs imports are used by individual modules as needed

// Implementation modules
pub mod chunk_bloom_filter;
pub mod chunked_backend; 
pub mod http_backend;
pub mod metadata;
#[cfg(feature = "mmap")]
pub mod mmap_backend;

// Public exports
pub use chunk_bloom_filter::ChunkBloomFilter;
pub use chunked_backend::{ChunkConfig, ChunkedMatrix, ChunkedProcessor};

// Memory mapping features
#[cfg(feature = "mmap")]
pub use mmap_backend::{BspcFile, DynamicElement, DynamicMatrix, MmapMatrix, SubmatrixView};

// HTTP backend features  
#[cfg(feature = "http")]
pub use http_backend::HttpMatrix;

// Metadata features
pub use metadata::{MetadataBuilder, MetadataView};

// Note: MatrixOperations for binsparse-rs Matrix types would require 
// orphan rule compliance. Users should wrap Matrix in their own type
// if they need MatrixOperations functionality.
