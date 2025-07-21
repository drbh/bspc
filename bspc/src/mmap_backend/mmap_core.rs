//! Core memory-mapped matrix structures and traits
//!
//! This module contains the fundamental types for working with memory-mapped sparse matrices.

use binsparse_rs::{Error, Result};
use bspc_core::{BspcHeader, DataType, MatrixFormat};
#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};
use std::{fs::File, path::Path};

/// Macro for safe array accessors
macro_rules! safe_array_accessor {
    ($name:ident, $field:ident, $len_field:ident, $type:ty) => {
        pub(crate) fn $name(&self) -> &[$type] {
            // SAFETY: Pointers and lengths validated during construction, mmap keeps memory alive
            unsafe { std::slice::from_raw_parts(self.$field, self.$len_field) }
        }
    };
}

/// Safely validate array size for u32 indices with overflow protection
pub(crate) fn validate_u32_array_size(byte_len: usize) -> Result<usize> {
    const U32_SIZE: usize = 4;

    // Check alignment
    if byte_len % U32_SIZE != 0 {
        return Err(Error::InvalidState("Array size not aligned to u32 size"));
    }

    // Use checked division
    let count = byte_len / U32_SIZE;

    // Check if array would be too large for safe indexing
    if count > isize::MAX as usize {
        return Err(Error::InvalidState("u32 array too large for safe indexing"));
    }

    // Verify we can multiply back without overflow
    count.checked_mul(U32_SIZE).ok_or(Error::InvalidState(
        "u32 array size calculation would overflow",
    ))?;

    Ok(count)
}

/// Helper function to create a typed slice from bytes with validation
fn create_typed_slice<T>(bytes: &[u8]) -> Result<&[T]>
where
    T: Copy + 'static,
{
    // Check size alignment
    let element_size = std::mem::size_of::<T>();
    if bytes.len() % element_size != 0 {
        return Err(Error::InvalidState(
            "Array size not aligned to element size",
        ));
    }

    // Check pointer alignment
    if (bytes.as_ptr() as usize) % std::mem::align_of::<T>() != 0 {
        return Err(Error::InvalidState("Array not properly aligned"));
    }

    let len = bytes.len() / element_size;

    // SAFETY:
    // 1. bytes is a valid slice with proper bounds
    // 2. Alignment was verified above
    // 3. Length calculation is correct
    // 4. T is Copy so no drop issues
    // 5. Lifetime is tied to input bytes
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, len) })
}

/// Helper function for u32 slices with additional validation
fn create_u32_slice(bytes: &[u8]) -> Result<&[u32]> {
    let len = validate_u32_array_size(bytes.len())?;
    let slice = create_typed_slice::<u32>(bytes)?;
    if slice.len() != len {
        return Err(Error::InvalidState("U32 slice length mismatch"));
    }
    Ok(slice)
}

/// Local trait for mmap-specific matrix element operations
///
/// This trait provides mmap-specific functionality like ArrayValue conversion
/// and byte serialization that builds on top of bspc_core::MatrixElement.
pub trait MatrixElement: bspc_core::MatrixElement + Send + Sync + 'static {
    /// Convert to ArrayValue for binsparse_rs compatibility
    fn to_array_value(self) -> binsparse_rs::array::ArrayValue;
    /// Read from bytes in little-endian format
    fn from_le_bytes(bytes: &[u8]) -> Result<Self>;
    /// Write to bytes in little-endian format
    fn to_le_bytes(self) -> Vec<u8>;
}

/// Macro to implement mmap-specific MatrixElement for primitive types
macro_rules! impl_mmap_matrix_element {
    ($type:ty, $array_variant:ident) => {
        impl MatrixElement for $type {
            fn to_array_value(self) -> binsparse_rs::array::ArrayValue {
                binsparse_rs::array::ArrayValue::$array_variant(self)
            }

            fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
                const SIZE: usize = std::mem::size_of::<$type>();
                if bytes.len() < SIZE {
                    return Err(Error::ConversionError(concat!(
                        "Insufficient bytes for ",
                        stringify!($type)
                    )));
                }
                let array: [u8; SIZE] = bytes[0..SIZE].try_into().map_err(|_| {
                    Error::ConversionError(concat!(
                        "Failed to convert bytes to ",
                        stringify!($type),
                        " array"
                    ))
                })?;
                Ok(<$type>::from_le_bytes(array))
            }

            fn to_le_bytes(self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        }
    };
}

// Implement mmap-specific MatrixElement for standard types
impl_mmap_matrix_element!(f32, Float32);
impl_mmap_matrix_element!(f64, Float64);
impl_mmap_matrix_element!(i32, Int32);
impl_mmap_matrix_element!(i64, Int64);
impl_mmap_matrix_element!(u32, UInt32);
impl_mmap_matrix_element!(u64, UInt64);

/// Memory-mapped matrix container that owns the memory mapping
/// and provides access to arrays using raw pointers with proper lifetime management
#[cfg(feature = "mmap")]
pub struct MmapMatrix<T: MatrixElement> {
    pub(crate) _mmap: Mmap, // Keep the mmap alive
    pub header: BspcHeader,
    pub(crate) values: *const T,
    pub(crate) values_len: usize,
    pub(crate) row_indices: *const u32,
    pub(crate) row_indices_len: usize,
    pub(crate) col_indices: *const u32,
    pub(crate) col_indices_len: usize,
    pub(crate) chunk_bloom_filter: crate::chunk_bloom_filter::ChunkBloomFilter,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

// SAFETY: MmapMatrix is safe to Send between threads because:
// 1. The raw pointers point to memory-mapped read-only data backed by _mmap
// 2. The _mmap field ensures the memory mapping stays alive
// 3. The data is immutable once created
// 4. T: MatrixElement already requires Send + Sync
#[cfg(feature = "mmap")]
unsafe impl<T: MatrixElement> Send for MmapMatrix<T> {}

// SAFETY: MmapMatrix is safe to share between threads (Sync) because:
// 1. All access methods are read-only
// 2. The underlying memory-mapped data is immutable
// 3. Raw pointers point to stable memory backed by the file system
// 4. No interior mutability is used
// 5. T: MatrixElement already requires Send + Sync
#[cfg(feature = "mmap")]
unsafe impl<T: MatrixElement> Sync for MmapMatrix<T> {}

#[cfg(feature = "mmap")]
impl<T: MatrixElement> MmapMatrix<T> {
    /// Load a matrix from a .bspc file using memory mapping
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path).map_err(|_| Error::IoError("Failed to open file"))?;

        // SAFETY: Read-only memory mapping with proper validation
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|_| Error::IoError("Failed to memory map file"))?
        };

        // Validate file size and header
        if mmap.len() < BspcHeader::SIZE {
            return Err(Error::InvalidState("File too small for header"));
        }

        let header = BspcHeader::from_bytes(&mmap[0..BspcHeader::SIZE])
            .map_err(|_| Error::InvalidState("Invalid BSPC header format"))?;

        // Validate and calculate offsets first (before creating slices)
        let values_start = header.values_offset as usize;
        let values_end = values_start
            .checked_add(header.values_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in values calculation",
            ))?;
        let row_indices_start = header.indices_0_offset as usize;
        let row_indices_end = row_indices_start
            .checked_add(header.indices_0_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in row indices calculation",
            ))?;
        let col_indices_start = header.indices_1_offset as usize;
        let col_indices_end = col_indices_start
            .checked_add(header.indices_1_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in column indices calculation",
            ))?;

        // Validate bounds
        if values_end > mmap.len() || row_indices_end > mmap.len() || col_indices_end > mmap.len() {
            return Err(Error::InvalidState("Arrays extend beyond file"));
        }

        // Load bloom filter or create if not present
        let loaded_bloom_filter = header
            .chunk_bloom_filter_region()
            .and_then(|(offset, size)| {
                let start = offset as usize;
                let end = start + size as usize;
                (end <= mmap.len()).then(|| &mmap[start..end])
            })
            .and_then(|data| crate::chunk_bloom_filter::ChunkBloomFilter::deserialize(data).ok());

        // Create the struct with mmap moved (bloom filter will be set later)
        let mut result = Self {
            _mmap: mmap,
            header,
            values: std::ptr::null(),
            values_len: 0,
            row_indices: std::ptr::null(),
            row_indices_len: 0,
            col_indices: std::ptr::null(),
            col_indices_len: 0,
            chunk_bloom_filter: crate::chunk_bloom_filter::ChunkBloomFilter::new(
                header.nrows as usize,
                100_000,
            ), // temporary, will be set properly
            _phantom: std::marker::PhantomData,
        };

        // Now create slices from the owned mmap
        let values_bytes = &result._mmap[values_start..values_end];
        let row_indices_bytes = &result._mmap[row_indices_start..row_indices_end];
        let col_indices_bytes = &result._mmap[col_indices_start..col_indices_end];

        // Validate alignment
        if (values_bytes.as_ptr() as usize) % std::mem::align_of::<T>() != 0 {
            return Err(Error::InvalidState("Values array not properly aligned"));
        }
        if (row_indices_bytes.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(Error::InvalidState(
                "Row indices array not properly aligned",
            ));
        }
        if (col_indices_bytes.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(Error::InvalidState(
                "Column indices array not properly aligned",
            ));
        }

        // Create typed slices
        let values = create_typed_slice::<T>(values_bytes)?;
        let row_indices = create_u32_slice(row_indices_bytes)?;
        let col_indices = create_u32_slice(col_indices_bytes)?;

        // Validate array consistency
        if values.len() != row_indices.len() || values.len() != col_indices.len() {
            return Err(Error::InvalidState("Array lengths don't match"));
        }
        if values.len() != header.nnz as usize {
            return Err(Error::InvalidState("Array length doesn't match nnz"));
        }

        // Set the pointers
        result.values = values.as_ptr();
        result.values_len = values.len();
        result.row_indices = row_indices.as_ptr();
        result.row_indices_len = row_indices.len();
        result.col_indices = col_indices.as_ptr();
        result.col_indices_len = col_indices.len();

        // Set the bloom filter - use loaded one or create from data
        result.chunk_bloom_filter = if let Some(bloom_filter) = loaded_bloom_filter {
            bloom_filter
        } else {
            // Create bloom filter from the matrix data
            let mut bloom_filter =
                crate::chunk_bloom_filter::ChunkBloomFilter::new(header.nrows as usize, 100_000);

            // Collect unique rows efficiently
            let mut unique_rows = Vec::new();
            let mut prev_row = None;
            for &row in row_indices {
                if prev_row != Some(row) {
                    unique_rows.push(row as usize);
                    prev_row = Some(row);
                }
            }

            bloom_filter.bulk_insert_sorted(&unique_rows);
            bloom_filter
        };

        Ok(result)
    }

    // Matrix dimensions - direct header access
    pub fn nrows(&self) -> usize {
        self.header.nrows as usize
    }
    pub fn ncols(&self) -> usize {
        self.header.ncols as usize
    }
    pub fn nnz(&self) -> usize {
        self.header.nnz as usize
    }

    safe_array_accessor!(values, values, values_len, T);
    safe_array_accessor!(row_indices, row_indices, row_indices_len, u32);
    safe_array_accessor!(col_indices, col_indices, col_indices_len, u32);

    // Simple accessors
    pub fn chunk_bloom_filter(&self) -> &crate::chunk_bloom_filter::ChunkBloomFilter {
        &self.chunk_bloom_filter
    }
    pub fn format(&self) -> MatrixFormat {
        MatrixFormat::from_u8(self.header.format_type).unwrap_or(MatrixFormat::Coo)
    }
    pub fn data_type(&self) -> DataType {
        DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64)
    }
}
