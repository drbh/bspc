//! Storage backend and processing traits for BSPC specification
//!
//! This module defines the abstract interfaces for storage backends
//! and chunk processing. These are pure interfaces with no implementations.

use crate::ChunkMetadata;

/// Trait for backends that can store sparse matrix data
///
/// This trait provides a minimal interface for accessing the underlying
/// byte data of a storage backend, regardless of how it's implemented
/// (memory-mapped files, HTTP ranges, in-memory buffers, etc.).
pub trait StorageBackend {
    /// Get a slice of the underlying data
    ///
    /// Returns a reference to the raw bytes stored by this backend.
    /// The slice may represent the entire dataset or a portion of it.
    fn as_slice(&self) -> &[u8];

    /// Get the size of the data in bytes
    ///
    /// Returns the total number of bytes available from this backend.
    /// Default implementation uses the slice length.
    fn size(&self) -> usize {
        self.as_slice().len()
    }
}

/// Trait for chunked matrix processing without allocations
///
/// This trait allows processing of large matrices in chunks without
/// requiring the entire matrix to be loaded into memory at once.
pub trait ChunkProcessor {
    /// The element type this processor works with
    type Element;

    /// Error type for processing operations
    type Error;

    /// Process a chunk of the matrix
    ///
    /// Called for each chunk of the matrix. The processor should
    /// extract and process the relevant data from the chunk.
    fn process_chunk(
        &mut self,
        chunk_data: &[u8],
        metadata: ChunkMetadata,
    ) -> Result<(), Self::Error>;

    /// Finalize processing and return results
    ///
    /// Called after all chunks have been processed. Should return
    /// the final result of the processing operation.
    fn finalize(self) -> Result<Self::Element, Self::Error>;
}

/// Trait for matrices that can be accessed in chunks
///
/// This trait provides an interface for matrices that support
/// chunked access patterns, allowing for memory-efficient processing
/// of large datasets.
pub trait Chunkable {
    /// The element type stored in this matrix
    type Element;

    /// Error type for chunking operations  
    type Error;

    /// Get the number of chunks in this matrix
    fn chunk_count(&self) -> usize;

    /// Get metadata for a specific chunk
    ///
    /// Returns the metadata describing the specified chunk,
    /// including its row range and data layout information.
    fn chunk_metadata(&self, chunk_index: usize) -> Result<ChunkMetadata, Self::Error>;

    /// Process chunks with a given processor
    ///
    /// Iterates through all chunks in the matrix and processes them
    /// with the provided processor. This allows for streaming processing
    /// of large matrices without loading everything into memory.
    fn process_chunks<P: ChunkProcessor<Element = Self::Element, Error = Self::Error>>(
        &self,
        processor: P,
    ) -> Result<Self::Element, Self::Error>;
}
