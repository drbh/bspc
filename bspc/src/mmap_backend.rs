//! Memory-mapped file backend for .bspc files
//!
//! This module implements a memory-mapped storage backend that can read and write
//! sparse matrices to/from .bspc files using memory mapping for efficient access.

use binsparse_rs::{array::ArrayValue, Error, Result};
use bspc_core::{BspcHeader, DataType, MatrixFormat};
#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};

/// Safely validate array size for u32 indices with overflow protection
fn validate_u32_array_size(byte_len: usize) -> Result<usize> {
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

/// Dynamic matrix that can hold any element type
pub enum DynamicMatrix {
    F32(MmapMatrix<f32>),
    F64(MmapMatrix<f64>),
    I32(MmapMatrix<i32>),
    I64(MmapMatrix<i64>),
    U32(MmapMatrix<u32>),
    U64(MmapMatrix<u64>),
}

impl DynamicMatrix {
    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        match self {
            DynamicMatrix::F32(m) => m.nrows(),
            DynamicMatrix::F64(m) => m.nrows(),
            DynamicMatrix::I32(m) => m.nrows(),
            DynamicMatrix::I64(m) => m.nrows(),
            DynamicMatrix::U32(m) => m.nrows(),
            DynamicMatrix::U64(m) => m.nrows(),
        }
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        match self {
            DynamicMatrix::F32(m) => m.ncols(),
            DynamicMatrix::F64(m) => m.ncols(),
            DynamicMatrix::I32(m) => m.ncols(),
            DynamicMatrix::I64(m) => m.ncols(),
            DynamicMatrix::U32(m) => m.ncols(),
            DynamicMatrix::U64(m) => m.ncols(),
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        match self {
            DynamicMatrix::F32(m) => m.nnz(),
            DynamicMatrix::F64(m) => m.nnz(),
            DynamicMatrix::I32(m) => m.nnz(),
            DynamicMatrix::I64(m) => m.nnz(),
            DynamicMatrix::U32(m) => m.nnz(),
            DynamicMatrix::U64(m) => m.nnz(),
        }
    }

    /// Get matrix format
    pub fn format(&self) -> MatrixFormat {
        match self {
            DynamicMatrix::F32(m) => m.format(),
            DynamicMatrix::F64(m) => m.format(),
            DynamicMatrix::I32(m) => m.format(),
            DynamicMatrix::I64(m) => m.format(),
            DynamicMatrix::U32(m) => m.format(),
            DynamicMatrix::U64(m) => m.format(),
        }
    }

    /// Get data type
    pub fn data_type(&self) -> DataType {
        match self {
            DynamicMatrix::F32(m) => m.data_type(),
            DynamicMatrix::F64(m) => m.data_type(),
            DynamicMatrix::I32(m) => m.data_type(),
            DynamicMatrix::I64(m) => m.data_type(),
            DynamicMatrix::U32(m) => m.data_type(),
            DynamicMatrix::U64(m) => m.data_type(),
        }
    }

    /// Get element at specific position
    pub fn get_element(&self, row: usize, col: usize) -> Result<Option<ArrayValue>> {
        match self {
            DynamicMatrix::F32(m) => m.get_element(row, col),
            DynamicMatrix::F64(m) => m.get_element(row, col),
            DynamicMatrix::I32(m) => m.get_element(row, col),
            DynamicMatrix::I64(m) => m.get_element(row, col),
            DynamicMatrix::U32(m) => m.get_element(row, col),
            DynamicMatrix::U64(m) => m.get_element(row, col),
        }
    }

    /// Get row view iterator with zero-copy access
    pub fn row_view(
        &self,
        row: usize,
    ) -> Result<Box<dyn Iterator<Item = (usize, ArrayValue)> + '_>> {
        match self {
            DynamicMatrix::F32(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::F64(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I32(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I64(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U32(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U64(m) => {
                let iter = m
                    .row_view(row)?
                    .map(|(col, val)| (col, val.to_array_value()));
                Ok(Box::new(iter))
            }
        }
    }

    /// Get column view iterator with zero-copy access
    pub fn col_view(
        &self,
        col: usize,
    ) -> Result<Box<dyn Iterator<Item = (usize, ArrayValue)> + '_>> {
        match self {
            DynamicMatrix::F32(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::F64(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I32(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I64(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U32(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U64(m) => {
                let iter = m
                    .col_view(col)?
                    .map(|(row, val)| (row, val.to_array_value()));
                Ok(Box::new(iter))
            }
        }
    }

    /// Get iterator over all rows in the matrix
    pub fn rows(&self) -> DynamicMatrixRowIterator<'_> {
        DynamicMatrixRowIterator {
            matrix: self,
            current_row: 0,
            total_rows: self.nrows(),
        }
    }

    /// Get efficient row range iterator that processes multiple rows in a single pass
    pub fn row_range_view(
        &self,
        start_row: usize,
        end_row: usize,
    ) -> Result<Box<dyn Iterator<Item = (usize, usize, ArrayValue)> + '_>> {
        match self {
            DynamicMatrix::F32(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::F64(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I32(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::I64(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U32(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
            DynamicMatrix::U64(m) => {
                let iter = m
                    .row_range_view(start_row, end_row)?
                    .map(|(row, col, val)| (row, col, val.to_array_value()));
                Ok(Box::new(iter))
            }
        }
    }

    /// Get metadata bytes from the memory-mapped file
    /// Returns None if no metadata is present
    pub fn metadata_bytes(&self) -> Option<&[u8]> {
        match self {
            DynamicMatrix::F32(m) => m.metadata_bytes(),
            DynamicMatrix::F64(m) => m.metadata_bytes(),
            DynamicMatrix::I32(m) => m.metadata_bytes(),
            DynamicMatrix::I64(m) => m.metadata_bytes(),
            DynamicMatrix::U32(m) => m.metadata_bytes(),
            DynamicMatrix::U64(m) => m.metadata_bytes(),
        }
    }

    /// Get structured metadata view for fast label lookups
    pub fn metadata_view(&self) -> Result<Option<crate::metadata::MetadataView<'_>>> {
        match self {
            DynamicMatrix::F32(m) => m.metadata_view(),
            DynamicMatrix::F64(m) => m.metadata_view(),
            DynamicMatrix::I32(m) => m.metadata_view(),
            DynamicMatrix::I64(m) => m.metadata_view(),
            DynamicMatrix::U32(m) => m.metadata_view(),
            DynamicMatrix::U64(m) => m.metadata_view(),
        }
    }

    /// Get row label by index
    pub fn row_label(&self, row_idx: u32) -> Result<Option<&[u8]>> {
        match self {
            DynamicMatrix::F32(m) => m.row_label(row_idx),
            DynamicMatrix::F64(m) => m.row_label(row_idx),
            DynamicMatrix::I32(m) => m.row_label(row_idx),
            DynamicMatrix::I64(m) => m.row_label(row_idx),
            DynamicMatrix::U32(m) => m.row_label(row_idx),
            DynamicMatrix::U64(m) => m.row_label(row_idx),
        }
    }

    /// Get column label by index
    pub fn col_label(&self, col_idx: u32) -> Result<Option<&[u8]>> {
        match self {
            DynamicMatrix::F32(m) => m.col_label(col_idx),
            DynamicMatrix::F64(m) => m.col_label(col_idx),
            DynamicMatrix::I32(m) => m.col_label(col_idx),
            DynamicMatrix::I64(m) => m.col_label(col_idx),
            DynamicMatrix::U32(m) => m.col_label(col_idx),
            DynamicMatrix::U64(m) => m.col_label(col_idx),
        }
    }
}

/// Iterator over all rows in a DynamicMatrix
pub struct DynamicMatrixRowIterator<'a> {
    matrix: &'a DynamicMatrix,
    current_row: usize,
    total_rows: usize,
}

impl<'a> Iterator for DynamicMatrixRowIterator<'a> {
    type Item = (
        usize,
        Result<Box<dyn Iterator<Item = (usize, ArrayValue)> + 'a>>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            return None;
        }

        let row_index = self.current_row;
        self.current_row += 1;

        Some((row_index, self.matrix.row_view(row_index)))
    }
}

impl<'a> ExactSizeIterator for DynamicMatrixRowIterator<'a> {
    fn len(&self) -> usize {
        self.total_rows.saturating_sub(self.current_row)
    }
}

// Implement ChunkableMatrix for DynamicMatrix
impl crate::chunked_backend::ChunkableMatrix for DynamicMatrix {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn get_element(&self, row: usize, col: usize) -> Result<Option<ArrayValue>> {
        self.get_element(row, col)
    }
}

/// Trait for types that can be used as matrix elements
pub trait MatrixElement: Copy + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Convert to ArrayValue
    fn to_array_value(self) -> ArrayValue;
    /// Convert from f64 (for compatibility with existing storage)
    fn from_f64(value: f64) -> Self;
    /// Get the corresponding DataType
    fn data_type() -> DataType;
    /// Size in bytes
    fn size_bytes() -> usize;
    /// Read from bytes
    fn from_le_bytes(bytes: &[u8]) -> Result<Self>;
    /// Write to bytes
    fn to_le_bytes(self) -> Vec<u8>;

    /// Safely calculate array length from byte size with overflow protection
    fn validate_array_size(byte_len: usize) -> Result<usize> {
        let element_size = Self::size_bytes();

        // Check alignment first
        if byte_len % element_size != 0 {
            return Err(Error::InvalidState(
                "Array size not aligned to element size",
            ));
        }

        // Use checked division to avoid overflow
        let count = byte_len / element_size;

        // Check if the resulting array would be too large
        if count > isize::MAX as usize {
            return Err(Error::InvalidState("Array too large for safe indexing"));
        }

        // Additional check: ensure we can multiply back without overflow
        count
            .checked_mul(element_size)
            .ok_or(Error::InvalidState("Array size calculation would overflow"))?;

        Ok(count)
    }

    /// Safely calculate total byte size from element count
    fn checked_byte_size(count: usize) -> Result<usize> {
        let element_size = Self::size_bytes();
        count
            .checked_mul(element_size)
            .ok_or(Error::InvalidState("Byte size calculation would overflow"))
    }
}

impl MatrixElement for f32 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::Float32(self)
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn data_type() -> DataType {
        DataType::F32
    }
    fn size_bytes() -> usize {
        4
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::ConversionError("Insufficient bytes for f32"));
        }
        let array: [u8; 4] = bytes[0..4]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to f32 array"))?;
        Ok(f32::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl MatrixElement for f64 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::Float64(self)
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn data_type() -> DataType {
        DataType::F64
    }
    fn size_bytes() -> usize {
        8
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::ConversionError("Insufficient bytes for f64"));
        }
        let array: [u8; 8] = bytes[0..8]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to f64 array"))?;
        Ok(f64::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl MatrixElement for i32 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::Int32(self)
    }
    fn from_f64(value: f64) -> Self {
        value as i32
    }
    fn data_type() -> DataType {
        DataType::I32
    }
    fn size_bytes() -> usize {
        4
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::ConversionError("Insufficient bytes for i32"));
        }
        let array: [u8; 4] = bytes[0..4]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to i32 array"))?;
        Ok(i32::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl MatrixElement for i64 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::Int64(self)
    }
    fn from_f64(value: f64) -> Self {
        value as i64
    }
    fn data_type() -> DataType {
        DataType::I64
    }
    fn size_bytes() -> usize {
        8
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::ConversionError("Insufficient bytes for i64"));
        }
        let array: [u8; 8] = bytes[0..8]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to i64 array"))?;
        Ok(i64::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl MatrixElement for u32 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::UInt32(self)
    }
    fn from_f64(value: f64) -> Self {
        value as u32
    }
    fn data_type() -> DataType {
        DataType::U32
    }
    fn size_bytes() -> usize {
        4
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::ConversionError("Insufficient bytes for u32"));
        }
        let array: [u8; 4] = bytes[0..4]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to u32 array"))?;
        Ok(u32::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl MatrixElement for u64 {
    fn to_array_value(self) -> ArrayValue {
        ArrayValue::UInt64(self)
    }
    fn from_f64(value: f64) -> Self {
        value as u64
    }
    fn data_type() -> DataType {
        DataType::U64
    }
    fn size_bytes() -> usize {
        8
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::ConversionError("Insufficient bytes for u64"));
        }
        let array: [u8; 8] = bytes[0..8]
            .try_into()
            .map_err(|_| Error::ConversionError("Failed to convert bytes to u64 array"))?;
        Ok(u64::from_le_bytes(array))
    }
    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

/// Memory-mapped matrix container that owns the memory mapping
/// and provides access to arrays using raw pointers with proper lifetime management
#[cfg(feature = "mmap")]
pub struct MmapMatrix<T: MatrixElement> {
    _mmap: Mmap, // Keep the mmap alive
    pub header: BspcHeader,
    values: *const T,
    values_len: usize,
    row_indices: *const u32,
    row_indices_len: usize,
    col_indices: *const u32,
    col_indices_len: usize,
    _phantom: std::marker::PhantomData<T>,
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
        // SAFETY: Memory mapping a file is inherently unsafe as it bypasses Rust's
        // memory safety guarantees. However, this is safe because:
        // 1. We only create a read-only memory mapping
        // 2. The file handle ensures the mapping remains valid
        // 3. We validate the file size before accessing any data
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|_| Error::IoError("Failed to memory map file"))?
        };

        // Read and validate header
        if mmap.len() < BspcHeader::SIZE {
            return Err(Error::InvalidState("File too small for header"));
        }

        // Check alignment
        if (mmap.as_ptr() as usize) % std::mem::align_of::<BspcHeader>() != 0 {
            return Err(Error::InvalidState("Header not properly aligned"));
        }

        let header_bytes = &mmap[0..BspcHeader::SIZE];
        let header = BspcHeader::from_bytes(header_bytes)
            .map_err(|_| Error::InvalidState("Invalid BSPC header format"))?;

        // Extract data arrays from memory map with validation using checked arithmetic
        let values_start = header.values_offset as usize;
        let values_end = values_start
            .checked_add(header.values_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in values array calculation",
            ))?;

        if values_end > mmap.len() {
            return Err(Error::InvalidState("Values array extends beyond file"));
        }

        let values_bytes = &mmap[values_start..values_end];

        // Check alignment for T
        if (values_bytes.as_ptr() as usize) % std::mem::align_of::<T>() != 0 {
            return Err(Error::InvalidState("Values array not properly aligned"));
        }

        // Use safe array size validation
        let values_len = T::validate_array_size(values_bytes.len())?;

        // SAFETY: Creating a slice from raw parts is unsafe, but safe here because:
        // 1. values_bytes is a valid slice from the memory map with proper bounds checking
        // 2. Alignment was verified above for type T
        // 3. values_len was validated using T::validate_array_size()
        // 4. The pointer is derived from a valid memory mapping
        // 5. The lifetime is tied to the mmap which outlives this slice
        let values: &[T] =
            unsafe { std::slice::from_raw_parts(values_bytes.as_ptr() as *const T, values_len) };

        let row_indices_start = header.indices_0_offset as usize;
        let row_indices_end = row_indices_start
            .checked_add(header.indices_0_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in row indices array calculation",
            ))?;

        if row_indices_end > mmap.len() {
            return Err(Error::InvalidState("Row indices array extends beyond file"));
        }

        let row_indices_bytes = &mmap[row_indices_start..row_indices_end];
        if row_indices_bytes.len() % 4 != 0 {
            return Err(Error::InvalidState(
                "Row indices array size not multiple of 4",
            ));
        }

        // Check alignment for u32
        if (row_indices_bytes.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(Error::InvalidState(
                "Row indices array not properly aligned for u32",
            ));
        }

        let row_indices_len = validate_u32_array_size(row_indices_bytes.len())?;

        // SAFETY: Creating u32 slice from raw parts is safe because:
        // 1. row_indices_bytes bounds were validated against the memory map
        // 2. Alignment for u32 was verified above
        // 3. row_indices_len was validated using validate_u32_array_size()
        // 4. The pointer is derived from valid memory mapping data
        // 5. The lifetime is tied to the mmap which outlives this slice
        let row_indices: &[u32] = unsafe {
            std::slice::from_raw_parts(row_indices_bytes.as_ptr() as *const u32, row_indices_len)
        };

        let col_indices_start = header.indices_1_offset as usize;
        let col_indices_end = col_indices_start
            .checked_add(header.indices_1_size as usize)
            .ok_or(Error::InvalidState(
                "Integer overflow in column indices array calculation",
            ))?;

        if col_indices_end > mmap.len() {
            return Err(Error::InvalidState(
                "Column indices array extends beyond file",
            ));
        }

        let col_indices_bytes = &mmap[col_indices_start..col_indices_end];

        // Check alignment for u32
        if (col_indices_bytes.as_ptr() as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(Error::InvalidState(
                "Column indices array not properly aligned for u32",
            ));
        }

        let col_indices_len = validate_u32_array_size(col_indices_bytes.len())?;

        // SAFETY: Creating u32 slice from raw parts is safe because:
        // 1. col_indices_bytes bounds were validated against the memory map
        // 2. Alignment for u32 was verified above
        // 3. col_indices_len was validated using validate_u32_array_size()
        // 4. The pointer is derived from valid memory mapping data
        // 5. The lifetime is tied to the mmap which outlives this slice
        let col_indices: &[u32] = unsafe {
            std::slice::from_raw_parts(col_indices_bytes.as_ptr() as *const u32, col_indices_len)
        };

        // Validate array lengths match
        if values.len() != row_indices.len() || values.len() != col_indices.len() {
            return Err(Error::InvalidState("Array lengths don't match"));
        }

        if values.len() != header.nnz as usize {
            return Err(Error::InvalidState("Array length doesn't match nnz"));
        }

        // Store raw pointers and lengths instead of extending lifetimes
        let values_ptr = values.as_ptr();
        let values_len = values.len();
        let row_indices_ptr = row_indices.as_ptr();
        let row_indices_len = row_indices.len();
        let col_indices_ptr = col_indices.as_ptr();
        let col_indices_len = col_indices.len();

        Ok(Self {
            _mmap: mmap,
            header,
            values: values_ptr,
            values_len,
            row_indices: row_indices_ptr,
            row_indices_len,
            col_indices: col_indices_ptr,
            col_indices_len,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get matrix dimensions
    pub fn nrows(&self) -> usize {
        self.header.nrows as usize
    }

    pub fn ncols(&self) -> usize {
        self.header.ncols as usize
    }

    pub fn nnz(&self) -> usize {
        self.header.nnz as usize
    }

    /// Safe accessor for values array
    fn values(&self) -> &[T] {
        // SAFETY: Raw pointer access is safe because:
        // 1. Pointer and length were validated during construction
        // 2. The mmap keeps the memory alive for the entire lifetime of self
        // 3. Data is immutable and accessed read-only
        // 4. No concurrent modification is possible
        unsafe { std::slice::from_raw_parts(self.values, self.values_len) }
    }

    /// Safe accessor for row indices array
    fn row_indices(&self) -> &[u32] {
        // SAFETY: Raw pointer access is safe because:
        // 1. Pointer and length were validated during construction
        // 2. The mmap keeps the memory alive for the entire lifetime of self
        // 3. Data is immutable and accessed read-only
        // 4. No concurrent modification is possible
        unsafe { std::slice::from_raw_parts(self.row_indices, self.row_indices_len) }
    }

    /// Safe accessor for column indices array
    fn col_indices(&self) -> &[u32] {
        // SAFETY: Raw pointer access is safe because:
        // 1. Pointer and length were validated during construction
        // 2. The mmap keeps the memory alive for the entire lifetime of self
        // 3. Data is immutable and accessed read-only
        // 4. No concurrent modification is possible
        unsafe { std::slice::from_raw_parts(self.col_indices, self.col_indices_len) }
    }

    /// Get element at specific position with comprehensive bounds checking
    pub fn get_element(&self, row: usize, col: usize) -> Result<Option<ArrayValue>> {
        // Validate input coordinates against matrix dimensions
        if row >= self.nrows() {
            return Err(Error::InvalidState("Row index out of bounds"));
        }
        if col >= self.ncols() {
            return Err(Error::InvalidState("Column index out of bounds"));
        }

        let values = self.values();
        let row_indices = self.row_indices();
        let col_indices = self.col_indices();

        // Ensure all arrays have the same length (data integrity check)
        if values.len() != row_indices.len() || values.len() != col_indices.len() {
            return Err(Error::InvalidState(
                "Corrupted data: array lengths don't match",
            ));
        }

        // For COO format, search through all entries with bounds checking
        for i in 0..values.len() {
            let file_row = row_indices[i] as usize;
            let file_col = col_indices[i] as usize;

            // Validate indices from file against matrix dimensions
            if file_row >= self.nrows() {
                return Err(Error::InvalidState(
                    "Corrupted data: row index exceeds matrix dimensions",
                ));
            }
            if file_col >= self.ncols() {
                return Err(Error::InvalidState(
                    "Corrupted data: column index exceeds matrix dimensions",
                ));
            }

            if file_row == row && file_col == col {
                return Ok(Some(values[i].to_array_value()));
            }
        }
        Ok(None)
    }

    /// Get matrix format
    pub fn format(&self) -> MatrixFormat {
        MatrixFormat::from(self.header.format_type)
    }

    /// Get data type
    pub fn data_type(&self) -> DataType {
        DataType::from(self.header.data_type)
    }

    /// Get metadata bytes from the memory-mapped file
    /// Returns None if no metadata is present
    pub fn metadata_bytes(&self) -> Option<&[u8]> {
        if let Some((offset, size)) = self.header.metadata_region() {
            // Validate offset and size against memory map bounds
            let start = offset as usize;
            let end = start.checked_add(size as usize)?;

            if end <= self._mmap.len() {
                Some(&self._mmap[start..end])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get structured metadata view for fast label lookups
    pub fn metadata_view(&self) -> Result<Option<crate::metadata::MetadataView<'_>>> {
        if let Some(metadata_bytes) = self.metadata_bytes() {
            Ok(Some(crate::metadata::MetadataView::new(metadata_bytes)?))
        } else {
            Ok(None)
        }
    }

    /// Get row label by index
    pub fn row_label(&self, row_idx: u32) -> Result<Option<&[u8]>> {
        if let Some(metadata_bytes) = self.metadata_bytes() {
            // Parse header
            let header = crate::metadata::BspcMetadataHeader::from_bytes(metadata_bytes)?;

            // Parse row labels array
            if header.row_labels_size == 0 {
                return Ok(None);
            }

            let start = header.row_labels_offset as usize;
            let end = start + header.row_labels_size as usize;

            if end > metadata_bytes.len() {
                return Err(Error::InvalidState("Row labels extend beyond metadata"));
            }

            let labels_data = &metadata_bytes[start..end];
            let labels_array = crate::metadata::LabelArray::from_bytes(labels_data)?;

            if row_idx >= labels_array.count {
                return Err(Error::InvalidState("Row index out of bounds"));
            }

            let label_start = crate::metadata::LabelArray::HEADER_SIZE
                + (row_idx as usize * labels_array.stride as usize);
            let label_end = label_start + labels_array.stride as usize;

            if label_end > labels_data.len() {
                return Err(Error::InvalidState("Row label extends beyond data"));
            }

            Ok(Some(&labels_data[label_start..label_end]))
        } else {
            Ok(None)
        }
    }

    /// Get column label by index
    pub fn col_label(&self, col_idx: u32) -> Result<Option<&[u8]>> {
        if let Some(metadata_bytes) = self.metadata_bytes() {
            // Parse header
            let header = crate::metadata::BspcMetadataHeader::from_bytes(metadata_bytes)?;

            // Parse column labels array
            if header.col_labels_size == 0 {
                return Ok(None);
            }

            let start = header.col_labels_offset as usize;
            let end = start + header.col_labels_size as usize;

            if end > metadata_bytes.len() {
                return Err(Error::InvalidState("Column labels extend beyond metadata"));
            }

            let labels_data = &metadata_bytes[start..end];
            let labels_array = crate::metadata::LabelArray::from_bytes(labels_data)?;

            if col_idx >= labels_array.count {
                return Err(Error::InvalidState("Column index out of bounds"));
            }

            let label_start = crate::metadata::LabelArray::HEADER_SIZE
                + (col_idx as usize * labels_array.stride as usize);
            let label_end = label_start + labels_array.stride as usize;

            if label_end > labels_data.len() {
                return Err(Error::InvalidState("Column label extends beyond data"));
            }

            Ok(Some(&labels_data[label_start..label_end]))
        } else {
            Ok(None)
        }
    }
}

// Implement ChunkableMatrix for MmapMatrix
#[cfg(feature = "mmap")]
impl<T: MatrixElement> crate::chunked_backend::ChunkableMatrix for MmapMatrix<T> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn get_element(&self, row: usize, col: usize) -> Result<Option<ArrayValue>> {
        self.get_element(row, col)
    }
}

/// View for matrix subregions (simplified for production)
pub struct SubmatrixView<T: MatrixElement> {
    elements: HashMap<(usize, usize), T>,
}

impl<T: MatrixElement> SubmatrixView<T> {
    pub fn new(
        matrix: &MmapMatrix<T>,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Result<Self> {
        // Validate row and column ranges
        if rows.end > matrix.nrows() {
            return Err(Error::InvalidState("Row range exceeds matrix dimensions"));
        }
        if cols.end > matrix.ncols() {
            return Err(Error::InvalidState(
                "Column range exceeds matrix dimensions",
            ));
        }
        if rows.is_empty() || cols.is_empty() {
            return Err(Error::InvalidState("Empty range not allowed"));
        }

        let mut elements = HashMap::new();
        let values = matrix.values();
        let row_indices = matrix.row_indices();
        let col_indices = matrix.col_indices();

        for i in 0..values.len() {
            let file_row = row_indices[i] as usize;
            let file_col = col_indices[i] as usize;

            // Validate file indices
            if file_row >= matrix.nrows() {
                return Err(Error::InvalidState(
                    "Corrupted data: row index exceeds matrix dimensions",
                ));
            }
            if file_col >= matrix.ncols() {
                return Err(Error::InvalidState(
                    "Corrupted data: column index exceeds matrix dimensions",
                ));
            }

            if rows.contains(&file_row) && cols.contains(&file_col) {
                elements.insert((file_row - rows.start, file_col - cols.start), values[i]);
            }
        }

        Ok(Self { elements })
    }

    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        self.elements.get(&(row, col)).copied()
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        self.elements.insert((row, col), value);
        Ok(())
    }

    pub fn get_as_array_value(&self, row: usize, col: usize) -> Option<ArrayValue> {
        self.elements.get(&(row, col)).map(|v| v.to_array_value())
    }
}

// Add view methods to MmapMatrix
#[cfg(feature = "mmap")]
impl<T: MatrixElement> MmapMatrix<T> {
    /// Get row view with zero-copy iterator
    pub fn row_view(&self, row: usize) -> Result<impl Iterator<Item = (usize, &T)> + '_> {
        // Validate row index
        if row >= self.nrows() {
            return Err(Error::InvalidState("Row index out of bounds"));
        }

        let values = self.values();
        let row_indices = self.row_indices();
        let col_indices = self.col_indices();

        Ok((0..values.len()).filter_map(move |i| {
            let file_row = row_indices[i] as usize;
            let file_col = col_indices[i] as usize;

            // Filter for matching row and validate bounds
            if file_row == row && file_row < self.nrows() && file_col < self.ncols() {
                Some((file_col, &values[i]))
            } else {
                None
            }
        }))
    }

    /// Create a submatrix view
    pub fn submatrix_view(
        &self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
    ) -> Result<SubmatrixView<T>> {
        SubmatrixView::new(self, rows, cols)
    }

    /// Get column view with zero-copy iterator
    pub fn col_view(&self, col: usize) -> Result<impl Iterator<Item = (usize, &T)> + '_> {
        // Validate column index
        if col >= self.ncols() {
            return Err(Error::InvalidState("Column index out of bounds"));
        }

        let values = self.values();
        let row_indices = self.row_indices();
        let col_indices = self.col_indices();

        Ok((0..values.len()).filter_map(move |i| {
            let file_col = col_indices[i] as usize;
            let file_row = row_indices[i] as usize;

            // Filter for matching column and validate bounds
            if file_col == col && file_row < self.nrows() && file_col < self.ncols() {
                Some((file_row, &values[i]))
            } else {
                None
            }
        }))
    }

    /// Get efficient row range iterator that processes multiple rows in a single pass
    pub fn row_range_view(
        &self,
        start_row: usize,
        end_row: usize,
    ) -> Result<impl Iterator<Item = (usize, usize, &T)> + '_> {
        // Validate row range
        if start_row >= self.nrows() || end_row > self.nrows() || start_row >= end_row {
            return Err(Error::InvalidState("Invalid row range"));
        }

        let values = self.values();
        let row_indices = self.row_indices();
        let col_indices = self.col_indices();

        // Single pass through all elements, filtering for the row range
        Ok((0..values.len()).filter_map(move |i| {
            let file_row = row_indices[i] as usize;
            let file_col = col_indices[i] as usize;

            // Filter for rows in range and validate bounds
            if file_row >= start_row
                && file_row < end_row
                && file_row < self.nrows()
                && file_col < self.ncols()
            {
                Some((file_row, file_col, &values[i]))
            } else {
                None
            }
        }))
    }
}

/// File handle for .bspc files with metadata  
pub struct BspcFile {
    pub header: BspcHeader,
    pub path: std::path::PathBuf,
}

impl BspcFile {
    /// Open an existing .bspc file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();

        // Read and validate header
        let mut file = File::open(&path_buf).map_err(|_| Error::IoError("Failed to open file"))?;
        let mut header_bytes = [0u8; BspcHeader::SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|_| Error::IoError("Failed to read header"))?;

        let header = BspcHeader::from_bytes(&header_bytes)
            .map_err(|_| Error::InvalidState("Invalid BSPC header format"))?;

        Ok(Self {
            header,
            path: path_buf,
        })
    }

    /// Create a new .bspc file
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            header: BspcHeader::default(),
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Read matrix with mandatory bloom filter optimization
    #[cfg(feature = "mmap")]
    pub fn read_matrix_with_bloom_filter<P: AsRef<Path>>(
        path: P,
        config: crate::chunked_backend::ChunkConfig,
    ) -> Result<crate::chunked_backend::ChunkedMatrix<DynamicMatrix>> {
        // First, read just the header to determine the data type
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref).map_err(|_| Error::IoError("Failed to open file"))?;
        let mut header_bytes = [0u8; BspcHeader::SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|_| Error::IoError("Failed to read header"))?;

        let header = BspcHeader::from_bytes(&header_bytes)
            .map_err(|_| Error::InvalidState("Invalid BSPC header format"))?;

        // Dispatch to the correct type based on the header
        let dynamic_matrix = match DataType::from(header.data_type) {
            DataType::F32 => DynamicMatrix::F32(MmapMatrix::from_file(path_ref)?),
            DataType::F64 => DynamicMatrix::F64(MmapMatrix::from_file(path_ref)?),
            DataType::I32 => DynamicMatrix::I32(MmapMatrix::from_file(path_ref)?),
            DataType::I64 => DynamicMatrix::I64(MmapMatrix::from_file(path_ref)?),
            DataType::U32 => DynamicMatrix::U32(MmapMatrix::from_file(path_ref)?),
            DataType::U64 => DynamicMatrix::U64(MmapMatrix::from_file(path_ref)?),
        };

        // Wrap in ChunkedMatrix for bloom filter optimization
        Ok(crate::chunked_backend::ChunkedMatrix::new(
            dynamic_matrix,
            config,
        ))
    }

    /// Write sparse matrix streaming (complete implementation)
    pub fn write_sparse_matrix_streaming<T: MatrixElement, P: AsRef<Path>>(
        nrows: usize,
        ncols: usize,
        sparse_elements: &[(usize, usize, T)],
        config: crate::chunked_backend::ChunkConfig,
        filename: P,
    ) -> Result<()> {
        let path = filename.as_ref();
        let mut file = File::create(path).map_err(|_| Error::IoError("Failed to create file"))?;

        let nnz = sparse_elements.len();

        // Calculate offsets with proper alignment and overflow checking
        let header_size = BspcHeader::SIZE as u32;

        // Align values to proper boundary for T
        let _element_size = T::size_bytes();
        let alignment = std::mem::align_of::<T>();
        let values_offset = header_size
            .div_ceil(alignment as u32)
            .checked_mul(alignment as u32)
            .ok_or(Error::InvalidState(
                "Integer overflow in values_offset calculation",
            ))?;

        // Use safe multiplication for values_size
        let values_size: u32 = T::checked_byte_size(nnz)?
            .try_into()
            .map_err(|_| Error::InvalidState("Values size too large for u32"))?;

        // Align indices to 4-byte boundary for u32
        let indices_0_offset = (values_offset + values_size).div_ceil(4) * 4;

        // Use safe multiplication for indices size
        let indices_0_size: u32 = nnz
            .checked_mul(4)
            .ok_or(Error::InvalidState(
                "Indices size calculation would overflow",
            ))?
            .try_into()
            .map_err(|_| Error::InvalidState("Indices size too large for u32"))?;

        let indices_1_offset = (indices_0_offset + indices_0_size).div_ceil(4) * 4;
        let indices_1_size: u32 = nnz
            .checked_mul(4)
            .ok_or(Error::InvalidState(
                "Indices size calculation would overflow",
            ))?
            .try_into()
            .map_err(|_| Error::InvalidState("Indices size too large for u32"))?;

        // Create header
        let mut header = BspcHeader::new();
        header.nrows = nrows as u32;
        header.ncols = ncols as u32;
        header.nnz = nnz as u32;
        header.format_type = MatrixFormat::Coo as u8;
        header.data_type = T::data_type() as u8;
        header.values_offset = values_offset;
        header.values_size = values_size;
        header.indices_0_offset = indices_0_offset;
        header.indices_0_size = indices_0_size;
        header.indices_1_offset = indices_1_offset;
        header.indices_1_size = indices_1_size;
        header.pointers_offset = 0; // Not used for COO
        header.pointers_size = 0;

        // Create bloom filter (mandatory for all matrices)
        use bspc_core::bloom_filter::BloomFilter;
        let mut bloom_filter = BloomFilter::<32>::with_hash_count(config.bloom_hash_count);

        // Insert all row indices into the bloom filter
        for &(row, _, _) in sparse_elements {
            bloom_filter.insert(row);
        }

        println!(
            "Created bloom filter with {} hash functions for {} rows",
            bloom_filter.hash_count(),
            nrows
        );

        // Write header
        // SAFETY: Converting struct to byte slice is unsafe but safe here because:
        // 1. BspcHeader is #[repr(C)] with fixed layout
        // 2. We're taking a reference to a valid stack variable
        // 3. The size is exactly BspcHeader::SIZE bytes
        // 4. The struct contains only primitive types with known representation
        // 5. We only read from this slice, no mutation
        let header_bytes = unsafe {
            std::slice::from_raw_parts(&header as *const BspcHeader as *const u8, BspcHeader::SIZE)
        };
        file.write_all(header_bytes)
            .map_err(|_| Error::IoError("Failed to write header"))?;

        // Add padding to align values to 8-byte boundary
        let padding_size = (values_offset as usize)
            .checked_sub(BspcHeader::SIZE)
            .ok_or(Error::InvalidState(
                "values_offset smaller than header size",
            ))?;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        // Write values
        for &(_, _, value) in sparse_elements {
            let value_bytes = value.to_le_bytes();
            file.write_all(&value_bytes)
                .map_err(|_| Error::IoError("Failed to write values"))?;
        }

        // Add padding to align row indices to 4-byte boundary
        let current_pos = values_offset + values_size;
        let padding_size = indices_0_offset - current_pos;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size as usize];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        // Write row indices
        for &(row, _, _) in sparse_elements {
            let row_bytes = (row as u32).to_le_bytes();
            file.write_all(&row_bytes)
                .map_err(|_| Error::IoError("Failed to write row indices"))?;
        }

        // Add padding to align column indices to 4-byte boundary
        let current_pos = indices_0_offset + indices_0_size;
        let padding_size = indices_1_offset - current_pos;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size as usize];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        // Write column indices
        for &(_, col, _) in sparse_elements {
            let col_bytes = (col as u32).to_le_bytes();
            file.write_all(&col_bytes)
                .map_err(|_| Error::IoError("Failed to write column indices"))?;
        }

        file.flush()
            .map_err(|_| Error::IoError("Failed to flush file"))?;

        println!(
            "Written matrix {}x{} with {} non-zeros to {} (bloom filter hash count: {})",
            nrows,
            ncols,
            nnz,
            path.display(),
            config.bloom_hash_count
        );

        Ok(())
    }

    /// Write sparse matrix with structured metadata
    pub fn write_sparse_matrix_with_labels<T: MatrixElement, P: AsRef<Path>>(
        nrows: usize,
        ncols: usize,
        sparse_elements: &[(usize, usize, T)],
        row_labels: &[&[u8]],
        col_labels: &[&[u8]],
        label_stride: u32,
        _config: crate::chunked_backend::ChunkConfig,
        filename: P,
    ) -> Result<()> {
        let path = filename.as_ref();
        let mut file = File::create(path).map_err(|_| Error::IoError("Failed to create file"))?;

        let nnz = sparse_elements.len();

        // Calculate matrix data layout first
        let header_size = BspcHeader::SIZE as u32;
        let _element_size = T::size_bytes();
        let alignment = std::mem::align_of::<T>();
        let values_offset = header_size
            .div_ceil(alignment as u32)
            .checked_mul(alignment as u32)
            .ok_or(Error::InvalidState(
                "Integer overflow in values_offset calculation",
            ))?;

        let values_size: u32 = T::checked_byte_size(nnz)?
            .try_into()
            .map_err(|_| Error::InvalidState("Values size too large for u32"))?;

        let indices_0_offset = (values_offset + values_size).div_ceil(4) * 4;
        let indices_0_size: u32 = nnz
            .checked_mul(4)
            .ok_or(Error::InvalidState(
                "Indices size calculation would overflow",
            ))?
            .try_into()
            .map_err(|_| Error::InvalidState("Indices size too large for u32"))?;

        let indices_1_offset = (indices_0_offset + indices_0_size).div_ceil(4) * 4;
        let indices_1_size: u32 = nnz
            .checked_mul(4)
            .ok_or(Error::InvalidState(
                "Indices size calculation would overflow",
            ))?
            .try_into()
            .map_err(|_| Error::InvalidState("Indices size too large for u32"))?;

        let matrix_end = indices_1_offset + indices_1_size;

        // Build structured metadata
        let metadata = if !row_labels.is_empty() || !col_labels.is_empty() {
            let mut builder = crate::metadata::MetadataBuilder::new(label_stride);

            if !row_labels.is_empty() {
                builder = builder
                    .with_row_labels(row_labels.iter().map(|&label| label.to_vec()).collect())?;
            }

            if !col_labels.is_empty() {
                builder = builder
                    .with_col_labels(col_labels.iter().map(|&label| label.to_vec()).collect())?;
            }

            Some(builder.build()?)
        } else {
            None
        };

        // Calculate metadata layout
        let metadata_start = crate::metadata::align_to_8(matrix_end as usize) as u64;
        let metadata_size = metadata.as_ref().map(|m| m.len() as u64).unwrap_or(0);

        // Create header with metadata pointer
        let mut header = BspcHeader::new();
        header.nrows = nrows as u32;
        header.ncols = ncols as u32;
        header.nnz = nnz as u32;
        header.format_type = MatrixFormat::Coo as u8;
        header.data_type = T::data_type() as u8;
        header.values_offset = values_offset;
        header.values_size = values_size;
        header.indices_0_offset = indices_0_offset;
        header.indices_0_size = indices_0_size;
        header.indices_1_offset = indices_1_offset;
        header.indices_1_size = indices_1_size;
        header.pointers_offset = 0;
        header.pointers_size = 0;

        // Set metadata region in header
        if let Some(ref _metadata) = metadata {
            header.set_metadata_region(metadata_start, metadata_size);
        }

        // Write header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(&header as *const BspcHeader as *const u8, BspcHeader::SIZE)
        };
        file.write_all(header_bytes)
            .map_err(|_| Error::IoError("Failed to write header"))?;

        // Write matrix data (same as before)
        let padding_size = (values_offset as usize)
            .checked_sub(BspcHeader::SIZE)
            .ok_or(Error::InvalidState(
                "values_offset smaller than header size",
            ))?;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        // Write values
        for &(_, _, value) in sparse_elements {
            let value_bytes = value.to_le_bytes();
            file.write_all(&value_bytes)
                .map_err(|_| Error::IoError("Failed to write values"))?;
        }

        // Write row indices
        let current_pos = values_offset + values_size;
        let padding_size = indices_0_offset - current_pos;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size as usize];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        for &(row, _, _) in sparse_elements {
            let row_bytes = (row as u32).to_le_bytes();
            file.write_all(&row_bytes)
                .map_err(|_| Error::IoError("Failed to write row indices"))?;
        }

        // Write column indices
        let current_pos = indices_0_offset + indices_0_size;
        let padding_size = indices_1_offset - current_pos;
        if padding_size > 0 {
            let padding = vec![0u8; padding_size as usize];
            file.write_all(&padding)
                .map_err(|_| Error::IoError("Failed to write alignment padding"))?;
        }

        for &(_, col, _) in sparse_elements {
            let col_bytes = (col as u32).to_le_bytes();
            file.write_all(&col_bytes)
                .map_err(|_| Error::IoError("Failed to write column indices"))?;
        }

        // Write metadata if present
        if let Some(metadata) = metadata {
            // Add padding to align metadata to 8-byte boundary
            let current_pos = indices_1_offset + indices_1_size;
            let padding_size = metadata_start - current_pos as u64;
            if padding_size > 0 {
                let padding = vec![0u8; padding_size as usize];
                file.write_all(&padding)
                    .map_err(|_| Error::IoError("Failed to write metadata alignment padding"))?;
            }

            file.write_all(&metadata)
                .map_err(|_| Error::IoError("Failed to write metadata"))?;
        }

        file.flush()
            .map_err(|_| Error::IoError("Failed to flush file"))?;

        println!(
            "Written matrix {}x{} with {} non-zeros and {} metadata bytes to {}",
            nrows,
            ncols,
            nnz,
            metadata_size,
            path.display()
        );

        Ok(())
    }
}
