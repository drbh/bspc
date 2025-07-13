//! Core traits for sparse matrix operations

/// Trait for backends that can store sparse matrix data
pub trait StorageBackend {
    /// Get a slice of the underlying data
    fn as_slice(&self) -> &[u8];

    /// Get the size of the data in bytes
    fn size(&self) -> usize {
        self.as_slice().len()
    }
}

/// Trait for chunked matrix processing without allocations
pub trait ChunkProcessor {
    type Element;
    type Error;

    /// Process a chunk of the matrix
    fn process_chunk(
        &mut self,
        chunk_data: &[u8],
        metadata: crate::ChunkMetadata,
    ) -> Result<(), Self::Error>;

    /// Finalize processing and return results
    fn finalize(self) -> Result<Self::Element, Self::Error>;
}

/// Trait for matrices that can be accessed in chunks
pub trait Chunkable {
    type Element;
    type Error;

    /// Get the number of chunks
    fn chunk_count(&self) -> usize;

    /// Get metadata for a specific chunk
    fn chunk_metadata(&self, chunk_index: usize) -> Result<crate::ChunkMetadata, Self::Error>;

    /// Process chunks with a given processor
    fn process_chunks<P: ChunkProcessor<Element = Self::Element, Error = Self::Error>>(
        &self,
        processor: P,
    ) -> Result<Self::Element, Self::Error>;
}
