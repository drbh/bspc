#![no_std]

//! BSPC Core - Binary Sparse Matrix Format Definitions
//!
//! This crate provides core format definitions and traits for binary sparse
//! matrix storage

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub mod bloom_filter;
pub mod error;
pub mod format;
pub mod traits;

pub use bloom_filter::*;
pub use error::*;
pub use format::*;
pub use traits::*;

/// Core sparse matrix trait for format-agnostic access
pub trait SparseMatrix {
    type Element;

    /// Get an element at the specified position
    fn get_element(&self, row: usize, col: usize) -> Option<Self::Element>;

    /// Get matrix dimensions as (rows, cols)
    fn dimensions(&self) -> (usize, usize);

    /// Get number of non-zero elements
    fn nnz(&self) -> usize;
}

/// Extension trait for row/column operations (requires alloc)
#[cfg(feature = "alloc")]
pub trait MatrixOperations: SparseMatrix {
    /// Get all elements in a row
    fn get_row(&self, row_index: usize) -> Vec<Self::Element>;

    /// Get all elements in a column  
    fn get_col(&self, col_index: usize) -> Vec<Self::Element>;
}
