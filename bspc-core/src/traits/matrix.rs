//! Core matrix abstraction traits for BSPC specification
//!
//! This module defines the fundamental traits that all matrix implementations
//! must satisfy. These are pure interfaces with no concrete implementations.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::element::MatrixElement;

/// Core sparse matrix trait for format-agnostic access
///
/// This trait provides the minimal interface that all sparse matrix
/// implementations must provide, regardless of storage backend.
pub trait SparseMatrix {
    /// The element type stored in this matrix
    type Element: MatrixElement;

    /// Get an element at the specified position
    ///
    /// Returns `None` if the element is zero (not stored) or if the
    /// position is out of bounds.
    fn get_element(&self, row: usize, col: usize) -> Option<Self::Element>;

    /// Get matrix dimensions as (rows, cols)
    fn dimensions(&self) -> (usize, usize);

    /// Get number of non-zero elements stored
    fn nnz(&self) -> usize;
}

/// Extension trait for row/column operations (requires alloc feature)
///
/// This trait provides higher-level operations that require allocation.
/// Only available when the `alloc` feature is enabled.
#[cfg(feature = "alloc")]
pub trait MatrixOperations: SparseMatrix {
    /// Get all non-zero elements in a row
    ///
    /// Returns a vector of all non-zero elements in the specified row.
    /// Elements are returned in column order.
    fn get_row(&self, row_index: usize) -> Vec<Self::Element>;

    /// Get all non-zero elements in a column
    ///
    /// Returns a vector of all non-zero elements in the specified column.
    /// Elements are returned in row order.
    fn get_col(&self, col_index: usize) -> Vec<Self::Element>;
}
