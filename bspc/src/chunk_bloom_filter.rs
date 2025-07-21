//! Chunk-level bloom filter for efficient sparse matrix access
//!
//! This module provides chunk-level bloom filtering that can skip entire chunks
//! of sparse matrix data when they don't contain elements in the requested range.

use bspc_core::bloom_filter::BloomFilter64;
use rayon::prelude::*;
use std::sync::Mutex;
use std::vec::Vec;

/// Chunk-level bloom filter for efficient sparse matrix access
#[derive(Debug, Clone)]
pub struct ChunkBloomFilter {
    /// Bloom filters for each chunk
    chunk_filters: Vec<BloomFilter64>,
    /// Size of each chunk in rows
    chunk_size: usize,
    /// Total number of rows in the matrix
    total_rows: usize,
}

impl ChunkBloomFilter {
    /// Create a new chunk bloom filter
    pub fn new(total_rows: usize, chunk_size: usize) -> Self {
        let num_chunks = total_rows.div_ceil(chunk_size);
        let chunk_filters = (0..num_chunks)
            .into_par_iter()
            .map(|_| BloomFilter64::new(chunk_size))
            .collect();

        Self {
            chunk_filters,
            chunk_size,
            total_rows,
        }
    }

    /// Create with specific hash count
    pub fn with_hash_count(total_rows: usize, chunk_size: usize, hash_count: u8) -> Self {
        let num_chunks = total_rows.div_ceil(chunk_size);
        let chunk_filters = (0..num_chunks)
            .into_par_iter()
            .map(|_| BloomFilter64::with_hash_count(hash_count))
            .collect();

        Self {
            chunk_filters,
            chunk_size,
            total_rows,
        }
    }

    /// Insert a row into the appropriate chunk's bloom filter
    pub fn insert(&mut self, row: usize) {
        let chunk_idx = row / self.chunk_size;
        if let Some(filter) = self.chunk_filters.get_mut(chunk_idx) {
            filter.insert(row % self.chunk_size);
        }
    }

    /// Fast bulk insert using rayon with minimal allocations
    pub fn bulk_insert_sorted(&mut self, sorted_rows: &[usize]) {
        if sorted_rows.is_empty() {
            return;
        }

        // Use atomic counters to partition work by chunk without Vec allocations
        use std::sync::atomic::AtomicUsize;

        // Create atomic indices for each chunk to track processing ranges
        let _chunk_indices: Vec<AtomicUsize> = (0..self.chunk_filters.len())
            .map(|_| AtomicUsize::new(0))
            .collect();

        // Process rows in parallel chunks, each thread works on a portion
        let num_threads = rayon::current_num_threads();
        let _chunk_size = sorted_rows.len().div_ceil(num_threads);

        // Use rayon's parallel iteration over the filters themselves
        self.chunk_filters
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk_idx, filter)| {
                let chunk_start = chunk_idx * self.chunk_size;
                let chunk_end = (chunk_idx + 1) * self.chunk_size;

                // Find relevant rows for this chunk using binary search
                let start_pos = sorted_rows.partition_point(|&row| row < chunk_start);
                let end_pos = sorted_rows.partition_point(|&row| row < chunk_end);

                // Insert all rows for this chunk
                for &row in &sorted_rows[start_pos..end_pos] {
                    filter.insert(row % self.chunk_size);
                }
            });
    }

    /// Check if a row range might contain data
    pub fn may_contain_range(&self, start_row: usize, end_row: usize) -> Vec<usize> {
        let start_chunk = start_row / self.chunk_size;
        let end_chunk = end_row.saturating_sub(1) / self.chunk_size;

        let chunk_range: Vec<usize> = (start_chunk..=end_chunk)
            .filter(|&chunk_idx| chunk_idx < self.chunk_filters.len())
            .collect();

        chunk_range
            .into_par_iter()
            .filter_map(|chunk_idx| {
                let filter = &self.chunk_filters[chunk_idx];
                let chunk_start = chunk_idx * self.chunk_size;
                let chunk_end = ((chunk_idx + 1) * self.chunk_size).min(self.total_rows);

                // Check if any row in the intersecting range might exist
                let range_start = start_row.max(chunk_start);
                let range_end = end_row.min(chunk_end);

                for row in range_start..range_end {
                    if filter.contains(row % self.chunk_size) {
                        return Some(chunk_idx);
                    }
                }
                None
            })
            .collect()
    }

    /// Check if a specific row might contain data
    pub fn may_contain_row(&self, row: usize) -> bool {
        let chunk_idx = row / self.chunk_size;
        if let Some(filter) = self.chunk_filters.get(chunk_idx) {
            filter.contains(row % self.chunk_size)
        } else {
            false
        }
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        self.chunk_filters.len()
    }

    /// Get the serialized size in bytes
    pub fn serialized_size(&self) -> usize {
        // 4 bytes for chunk_size + 4 bytes for total_rows + 4 bytes for num_chunks
        // + 1 byte for hash_count per chunk + 8 bytes per chunk filter
        12 + self.chunk_filters.len() * 9
    }

    /// Serialize the chunk bloom filter
    pub fn serialize(&self) -> Vec<u8> {
        let buffer = Mutex::new(Vec::with_capacity(self.serialized_size()));

        // Write metadata
        {
            let mut buf = buffer.lock().unwrap();
            buf.extend_from_slice(&(self.chunk_size as u32).to_le_bytes());
            buf.extend_from_slice(&(self.total_rows as u32).to_le_bytes());
            buf.extend_from_slice(&(self.chunk_filters.len() as u32).to_le_bytes());
        }

        // Parallel serialization of chunk filters
        let chunk_data: Vec<_> = self
            .chunk_filters
            .par_iter()
            .map(|filter| {
                let mut chunk_bytes = Vec::with_capacity(9);
                chunk_bytes.push(filter.hash_count());
                chunk_bytes.extend_from_slice(filter.bits());
                chunk_bytes
            })
            .collect();

        // Sequential append to maintain order
        let mut buf = buffer.into_inner().unwrap();
        for chunk_bytes in chunk_data {
            buf.extend_from_slice(&chunk_bytes);
        }

        buf
    }

    /// Deserialize a chunk bloom filter
    pub fn deserialize(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 12 {
            return Err("Invalid chunk bloom filter data");
        }

        let chunk_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let total_rows = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let num_chunks = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let mut chunk_filters = Vec::with_capacity(num_chunks);
        let mut offset = 12;

        for _ in 0..num_chunks {
            if offset + 9 > data.len() {
                return Err("Invalid chunk bloom filter data");
            }

            let hash_count = data[offset];
            offset += 1;

            let mut bits = [0u8; 8];
            bits.copy_from_slice(&data[offset..offset + 8]);
            offset += 8;

            chunk_filters.push(BloomFilter64::from_bits(bits, hash_count));
        }

        Ok(Self {
            chunk_filters,
            chunk_size,
            total_rows,
        })
    }
}
