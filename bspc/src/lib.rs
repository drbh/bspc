//! BSPC - Binary Sparse Matrix format with memory mapping support
//!
//! This library provides efficient storage and access of sparse matrices using
//! memory mapping for zero-copy operations with bloom filter optimizations.

// Re-export core functionality
pub use bspc_core::*;

use binsparse_rs::{array::ArrayValue, matrix::Matrix};

// I/O and memory mapping features
pub mod chunked_backend;
#[cfg(feature = "mmap")]
pub mod mmap_backend;

pub use chunked_backend::{ChunkConfig, ChunkableMatrix, ChunkedMatrix, ChunkedProcessor};

// Memory mapping features (optional)
#[cfg(feature = "mmap")]
pub use mmap_backend::{BspcFile, DynamicMatrix, MatrixElement, MmapMatrix, SubmatrixView};

/// Extension trait to add row and column access methods to Matrix
pub trait MatrixExtensions<T: binsparse_rs::backend::StorageBackend> {
    fn get_row(&self, row_index: usize) -> binsparse_rs::Result<Vec<ArrayValue>>;
    fn get_col(&self, col_index: usize) -> binsparse_rs::Result<Vec<ArrayValue>>;
}

impl<T: binsparse_rs::backend::StorageBackend> MatrixExtensions<T> for Matrix<T> {
    fn get_row(&self, row_index: usize) -> binsparse_rs::Result<Vec<ArrayValue>> {
        if row_index >= self.nrows {
            return Ok(Vec::new());
        }

        let mut row_elements = Vec::new();
        for col in 0..self.ncols {
            if let Some(value) = self.get_element(row_index, col)? {
                row_elements.push(value);
            }
        }
        Ok(row_elements)
    }

    fn get_col(&self, col_index: usize) -> binsparse_rs::Result<Vec<ArrayValue>> {
        if col_index >= self.ncols {
            return Ok(Vec::new());
        }

        let mut col_elements = Vec::new();
        for row in 0..self.nrows {
            if let Some(value) = self.get_element(row, col_index)? {
                col_elements.push(value);
            }
        }
        Ok(col_elements)
    }
}
