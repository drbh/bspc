//! Chunked memory-mapped backend for resource-constrained sparse matrix access
//!
//! This module provides a generic chunked access layer over memory-mapped sparse matrices,
//! enabling efficient processing of large datasets within bounded memory constraints.

use binsparse_rs::prelude::*;
use bspc_core::SparseMatrix;

// Note: SparseMatrix implementation for binsparse-rs Matrix types
// is provided in a separate adapter module to comply with orphan rules

/// Configuration for chunked processing with chunk-level bloom filters
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum memory usage per chunk in MB
    pub memory_limit_mb: usize,
    /// Number of hash functions for bloom filters
    pub bloom_hash_count: u8,
    /// Size of each chunk in rows for bloom filtering
    pub chunk_size: usize,
}

impl ChunkConfig {
    /// Create config with memory limit
    pub fn with_memory_limit(memory_limit_mb: usize) -> Self {
        Self {
            memory_limit_mb,
            bloom_hash_count: 3,
            chunk_size: 100_000, // Default 100K rows per chunk
        }
    }

    /// Set bloom filter hash count
    pub fn with_bloom_hash_count(mut self, hash_count: u8) -> Self {
        self.bloom_hash_count = hash_count;
        self
    }

    /// Set chunk size in rows for bloom filtering
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Get chunk size in rows
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get chunk size in bytes (compatibility property)
    pub fn chunk_size_bytes(&self) -> usize {
        self.memory_limit_mb * 1024 * 1024
    }

    /// Calculate optimal chunk size based on matrix characteristics
    pub fn optimal_chunk_size(matrix_rows: usize, nnz: usize, typical_query_size: usize) -> usize {
        let density = nnz as f64 / matrix_rows as f64;
        let base_chunk_size = typical_query_size.max(1000) * 2; // At least 2x typical query, minimum 2000

        if density < 0.01 {
            // Very sparse - use smaller chunks for better filtering
            base_chunk_size.min(50_000)
        } else if density > 0.1 {
            // Dense - larger chunks are fine
            base_chunk_size.max(200_000)
        } else {
            base_chunk_size.clamp(50_000, 200_000)
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: 128,
            bloom_hash_count: 3,
            chunk_size: 100_000,
        }
    }
}

/// Chunked processor for stream processing
pub struct ChunkedProcessor<M: SparseMatrix> {
    matrix: M,
    config: ChunkConfig,
}

impl<M: SparseMatrix> ChunkedProcessor<M> {
    /// Create a new chunked processor
    pub fn new(matrix: M, config: ChunkConfig) -> Self {
        Self { matrix, config }
    }

    /// Get element from matrix
    pub fn get_element(&self, row: usize, col: usize) -> Option<M::Element> {
        self.matrix.get_element(row, col)
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.matrix.dimensions()
    }

    /// Get the chunk configuration
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }
}

/// Chunked matrix wrapper
pub struct ChunkedMatrix<M: SparseMatrix> {
    inner: ChunkedProcessor<M>,
}

impl<M: SparseMatrix> ChunkedMatrix<M> {
    /// Create a new chunked matrix
    pub fn new(matrix: M, config: ChunkConfig) -> Self {
        Self {
            inner: ChunkedProcessor::new(matrix, config),
        }
    }

    /// Get reference to the underlying matrix
    pub fn matrix(&self) -> &M {
        &self.inner.matrix
    }

    /// Create from file (compatibility method)
    #[cfg(feature = "mmap")]
    pub fn from_file<T: crate::mmap_backend::MatrixElement + bspc_core::MatrixElement, P: AsRef<std::path::Path>>(
        path: P,
        config: ChunkConfig,
    ) -> Result<ChunkedMatrix<crate::mmap_backend::MmapMatrix<T>>> {
        let mmap_matrix = crate::mmap_backend::MmapMatrix::from_file(path)?;
        Ok(ChunkedMatrix::new(mmap_matrix, config))
    }

    /// Get element from matrix
    pub fn get_element(&self, row: usize, col: usize) -> Option<M::Element> {
        self.inner.get_element(row, col)
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.inner.dimensions()
    }

    /// Random access benchmark (stub implementation)
    pub fn random_access<F>(
        &self,
        operations: usize,
        mut f: F,
    ) -> Result<(std::time::Duration, String)>
    where
        F: FnMut(usize, usize) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let (nrows, ncols) = self.dimensions();

        if nrows == 0 || ncols == 0 {
            return Ok((
                start.elapsed(),
                format!("Cannot perform random access on empty matrix ({nrows}x{ncols})"),
            ));
        }

        for i in 0..operations {
            let row = i % nrows;
            let col = i % ncols;
            f(row, col)?;
        }

        let duration = start.elapsed();
        Ok((duration, format!("Performed {operations} random accesses")))
    }

    /// Streaming access benchmark (stub implementation)
    pub fn streaming_access<F>(
        &self,
        operations: usize,
        _batch_size: usize,
        mut f: F,
    ) -> Result<(std::time::Duration, String)>
    where
        F: FnMut(usize, usize) -> Result<()>,
    {
        let start = std::time::Instant::now();
        let (nrows, ncols) = self.dimensions();

        if nrows == 0 || ncols == 0 {
            return Ok((
                start.elapsed(),
                format!("Cannot perform streaming access on empty matrix ({nrows}x{ncols})"),
            ));
        }

        for i in 0..operations {
            let row = i % nrows;
            let col = i % ncols;
            f(row, col)?;
        }

        let duration = start.elapsed();
        Ok((
            duration,
            format!("Performed {operations} streaming accesses"),
        ))
    }
}

// Special implementation for ChunkedMatrix<DynamicMatrix> to provide row/column iterators
impl ChunkedMatrix<crate::mmap_backend::DynamicMatrix> {
    /// Get row view iterator
    pub fn row_view(
        &self,
        row: usize,
    ) -> binsparse_rs::Result<Box<dyn Iterator<Item = (usize, binsparse_rs::array::ArrayValue)> + '_>>
    {
        self.inner
            .matrix
            .row_view(row)
            .map_err(|_| binsparse_rs::Error::InvalidState("Row view error"))
    }

    /// Get column view iterator
    pub fn col_view(
        &self,
        col: usize,
    ) -> binsparse_rs::Result<Box<dyn Iterator<Item = (usize, binsparse_rs::array::ArrayValue)> + '_>>
    {
        self.inner
            .matrix
            .col_view(col)
            .map_err(|_| binsparse_rs::Error::InvalidState("Column view error"))
    }

    /// Get iterator over all rows in the matrix
    pub fn rows(&self) -> ChunkedMatrixRowIterator<'_> {
        ChunkedMatrixRowIterator {
            matrix: &self.inner.matrix,
            current_row: 0,
            total_rows: self.inner.matrix.nrows(),
        }
    }

    /// Get iterator over a range of rows
    pub fn rows_range(&self, start: usize, end: usize) -> ChunkedMatrixRowIterator<'_> {
        let total_rows = self.inner.matrix.nrows();
        let end = end.min(total_rows);
        let start = start.min(end);

        ChunkedMatrixRowIterator {
            matrix: &self.inner.matrix,
            current_row: start,
            total_rows: end,
        }
    }

    /// Get optimized row range iterator for bulk processing (single pass through data)
    pub fn row_range_view(
        &self,
        start: usize,
        end: usize,
    ) -> binsparse_rs::Result<
        Box<dyn Iterator<Item = (usize, usize, binsparse_rs::array::ArrayValue)> + '_>,
    > {
        self.inner
            .matrix
            .row_range_view(start, end)
            .map_err(|_| binsparse_rs::Error::InvalidState("Row range view error"))
    }
}

/// Iterator over all rows in a ChunkedMatrix<DynamicMatrix>
pub struct ChunkedMatrixRowIterator<'a> {
    matrix: &'a crate::mmap_backend::DynamicMatrix,
    current_row: usize,
    total_rows: usize,
}

impl<'a> Iterator for ChunkedMatrixRowIterator<'a> {
    type Item = (
        usize,
        binsparse_rs::Result<
            Box<dyn Iterator<Item = (usize, binsparse_rs::array::ArrayValue)> + 'a>,
        >,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            return None;
        }

        let row_index = self.current_row;
        self.current_row += 1;

        let row_result = self
            .matrix
            .row_view(row_index)
            .map_err(|_| binsparse_rs::Error::InvalidState("Row view error"));

        Some((row_index, row_result))
    }
}

impl<'a> ExactSizeIterator for ChunkedMatrixRowIterator<'a> {
    fn len(&self) -> usize {
        self.total_rows.saturating_sub(self.current_row)
    }
}
