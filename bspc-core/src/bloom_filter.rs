//! Compact bloom filter implementation optimized for no_std environments
//!
//! This module provides fixed-size bloom filters.

/// Compact bloom filter with fixed-size bit array
#[derive(Debug, Clone, PartialEq)]
pub struct BloomFilter<const N: usize = 32> {
    /// Bit array (N bytes = N*8 bits)
    bits: [u8; N],
    /// Number of hash functions to use
    hash_count: u8,
}

impl<const N: usize> BloomFilter<N> {
    /// Create a new bloom filter optimized for expected element count
    pub fn new(expected_elements: usize) -> Self {
        // Optimal hash function count: k = (m/n) * ln(2)
        // where m = N*8 bits, n = expected_elements
        let optimal_k = if expected_elements > 0 {
            // Calculate optimal hash count: k = (m/n) * ln(2)
            // Using integer approximation: ln(2) ≈ 0.693 ≈ 693/1000
            let m = N * 8; // Total bits
            let k_times_1000 = (m * 693) / expected_elements; // k * 1000
            let k = k_times_1000.div_ceil(1000); // Ceiling division
            k as u8
        } else {
            3
        };

        // Clamp to reasonable range
        let hash_count = optimal_k.clamp(1, 8);

        Self {
            bits: [0; N],
            hash_count,
        }
    }

    /// Create a new bloom filter with specified hash function count
    pub const fn with_hash_count(hash_count: u8) -> Self {
        Self {
            bits: [0; N],
            hash_count,
        }
    }

    /// Insert a value into the bloom filter
    pub fn insert(&mut self, value: usize) {
        for i in 0..self.hash_count {
            let hash = self.hash_function(value, i);
            let bit_index = hash % (N * 8);
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;

            self.bits[byte_index] |= 1 << bit_offset;
        }
    }

    /// Check if a value might be in the set (may have false positives)
    pub fn contains(&self, value: usize) -> bool {
        for i in 0..self.hash_count {
            let hash = self.hash_function(value, i);
            let bit_index = hash % (N * 8);
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;

            if (self.bits[byte_index] & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear all bits in the filter
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Get the number of hash functions
    pub fn hash_count(&self) -> u8 {
        self.hash_count
    }

    /// Get the number of bits in the filter
    pub const fn bit_count() -> usize {
        N * 8
    }

    /// Get the bits array (for serialization)
    pub fn bits(&self) -> &[u8; N] {
        &self.bits
    }

    /// Create a bloom filter from raw bits and hash count (for deserialization)
    pub fn from_bits(bits: [u8; N], hash_count: u8) -> Self {
        Self { bits, hash_count }
    }

    /// Fast hash function combining FNV-1a with seed
    fn hash_function(&self, value: usize, seed: u8) -> usize {
        // Simple but effective hash function for bloom filters
        let mut hash = 2166136261u64; // FNV offset basis
        let value_bytes = value.to_le_bytes();

        // Hash the value bytes
        for &byte in &value_bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(16777619); // FNV prime
        }

        // Mix in the seed
        hash ^= seed as u64;
        hash = hash.wrapping_mul(16777619);

        hash as usize
    }
}

impl<const N: usize> Default for BloomFilter<N> {
    fn default() -> Self {
        Self::new(100) // Default to 100 expected elements
    }
}

/// Standard 32-byte (256-bit) bloom filter
pub type BloomFilter256 = BloomFilter<32>;

/// Compact 8-byte (64-bit) bloom filter
pub type BloomFilter64 = BloomFilter<8>;

/// Large 128-byte (1024-bit) bloom filter
pub type BloomFilter1024 = BloomFilter<128>;

/// Chunk metadata for chunked processing
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkMetadata {
    /// Starting row index for this chunk
    pub start_row: usize,
    /// Ending row index for this chunk (exclusive)
    pub end_row: usize,
    /// Number of non-zero elements in this chunk
    pub nnz: usize,
    /// Offset in the data file where this chunk starts
    pub data_offset: u64,
    /// Size of this chunk's data in bytes
    pub data_size: usize,
}

impl ChunkMetadata {
    /// Create new chunk metadata
    pub const fn new(
        start_row: usize,
        end_row: usize,
        nnz: usize,
        data_offset: u64,
        data_size: usize,
    ) -> Self {
        Self {
            start_row,
            end_row,
            nnz,
            data_offset,
            data_size,
        }
    }

    /// Check if a row index falls within this chunk
    pub fn contains_row(&self, row: usize) -> bool {
        row >= self.start_row && row < self.end_row
    }

    /// Get the number of rows in this chunk
    pub fn row_count(&self) -> usize {
        self.end_row - self.start_row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter256::new(10);

        // Insert some values
        filter.insert(42);
        filter.insert(100);
        filter.insert(500);

        // Check that inserted values are found
        assert!(filter.contains(42));
        assert!(filter.contains(100));
        assert!(filter.contains(500));

        // Clear and verify
        filter.clear();
        assert!(!filter.contains(42));
    }
}
