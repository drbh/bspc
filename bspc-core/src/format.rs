//! Binary sparse matrix format definitions
//!
//! This module defines the core data structures and binary format
//! specifications for .bspc files.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::mem::size_of;

/// Standard header for .bspc files (u64 based, supports large datasets)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BspcHeader {
    /// Magic bytes: "BSPC"
    pub magic: [u8; 4],
    /// Format version
    pub version: u8,
    /// Matrix format type (COO=0, CSR=1, CSC=2, etc.)
    pub format_type: u8,
    /// Data type (f32=0, f64=1, i32=2, etc.)
    pub data_type: u8,
    /// Structure flags (symmetric, upper_tri, etc.)
    pub structure_flags: u8,
    /// Number of rows (u64 for large matrices)
    pub nrows: u64,
    /// Number of columns (u64 for large matrices)
    pub ncols: u64,
    /// Number of non-zero elements (u64 for large datasets)
    pub nnz: u64,
    /// Offset to values array (u64 for large files)
    pub values_offset: u64,
    /// Size of values array in bytes (u64 for large arrays)
    pub values_size: u64,
    /// Offset to first index array (u64 for large files)
    pub indices_0_offset: u64,
    /// Size of first index array in bytes (u64 for large arrays)
    pub indices_0_size: u64,
    /// Offset to second index array (u64 for large files)
    pub indices_1_offset: u64,
    /// Size of second index array in bytes (u64 for large arrays)
    pub indices_1_size: u64,
    /// Offset to pointers array (u64 for large files)
    pub pointers_offset: u64,
    /// Size of pointers array in bytes (u64 for large arrays)
    pub pointers_size: u64,
    /// Offset to metadata region (u64)
    pub metadata_offset: u64,
    /// Size of metadata region in bytes (u64)
    pub metadata_size: u64,
    /// Offset to chunk bloom filter region (u64)
    pub bloom_filter_offset: u64,
    /// Size of chunk bloom filter region in bytes (u64)
    pub bloom_filter_size: u64,
    /// Reserved bytes for future extensions
    pub reserved: [u8; 32],
}

impl BspcHeader {
    /// Magic bytes for .bspc files
    pub const MAGIC: [u8; 4] = *b"BSPC";

    /// Current format version
    pub const VERSION: u8 = 1;

    /// Size of the header in bytes
    pub const SIZE: usize = size_of::<Self>();

    /// Create a new header with default values
    pub const fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            format_type: 0,
            data_type: 0,
            structure_flags: 0,
            nrows: 0,
            ncols: 0,
            nnz: 0,
            values_offset: 0,
            values_size: 0,
            indices_0_offset: 0,
            indices_0_size: 0,
            indices_1_offset: 0,
            indices_1_size: 0,
            pointers_offset: 0,
            pointers_size: 0,
            metadata_offset: 0,
            metadata_size: 0,
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
            reserved: [0; 32],
        }
    }

    /// Get metadata region offset and size
    pub fn metadata_region(&self) -> Option<(u64, u64)> {
        if self.metadata_offset == 0 || self.metadata_size == 0 {
            None
        } else {
            Some((self.metadata_offset, self.metadata_size))
        }
    }

    /// Get chunk bloom filter region offset and size
    pub fn chunk_bloom_filter_region(&self) -> Option<(u64, u64)> {
        if self.bloom_filter_offset == 0 || self.bloom_filter_size == 0 {
            None
        } else {
            Some((self.bloom_filter_offset, self.bloom_filter_size))
        }
    }

    /// Set chunk bloom filter region offset and size
    pub fn set_chunk_bloom_filter_region(&mut self, offset: u64, size: u64) {
        self.bloom_filter_offset = offset;
        self.bloom_filter_size = size;
    }

    /// Set metadata region offset and size
    pub fn set_metadata_region(&mut self, offset: u64, size: u64) {
        self.metadata_offset = offset;
        self.metadata_size = size;
    }

    /// Validate header structure integrity
    pub fn is_valid(&self) -> bool {
        // Check magic bytes
        if self.magic != Self::MAGIC {
            return false;
        }

        // Check version
        if self.version != Self::VERSION {
            return false;
        }

        // Basic sanity checks on dimensions
        if self.nrows == 0 || self.ncols == 0 {
            return false;
        }

        // Check that nnz doesn't exceed matrix capacity
        if let Some(max_elements) = self.nrows.checked_mul(self.ncols) {
            if self.nnz > max_elements {
                return false;
            }
        } else {
            // Overflow in matrix dimensions
            return false;
        }

        true
    }

    /// Safely read header from bytes with validation
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Validate magic bytes first
        if bytes[0..4] != Self::MAGIC {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Read header fields
        let mut header = Self::new();
        header.magic.copy_from_slice(&bytes[0..4]);
        header.version = bytes[4];
        header.format_type = bytes[5];
        header.data_type = bytes[6];
        header.structure_flags = bytes[7];

        // Helper to read u64 from bytes
        let read_u64 = |offset: usize| -> u64 {
            u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ])
        };

        // Read u64 fields
        header.nrows = read_u64(8);
        header.ncols = read_u64(16);
        header.nnz = read_u64(24);
        header.values_offset = read_u64(32);
        header.values_size = read_u64(40);
        header.indices_0_offset = read_u64(48);
        header.indices_0_size = read_u64(56);
        header.indices_1_offset = read_u64(64);
        header.indices_1_size = read_u64(72);
        header.pointers_offset = read_u64(80);
        header.pointers_size = read_u64(88);
        header.metadata_offset = read_u64(96);
        header.metadata_size = read_u64(104);
        header.bloom_filter_offset = read_u64(112);
        header.bloom_filter_size = read_u64(120);

        // Copy reserved bytes
        header.reserved.copy_from_slice(&bytes[128..160]);

        // Validate the header structure
        if !header.is_valid() {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Additional validation
        if header.nrows == 0 || header.ncols == 0 {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Check for potential integer overflow in matrix size
        let total_elements = header
            .nrows
            .checked_mul(header.ncols)
            .ok_or(crate::error::BspcError::InvalidHeader)?;

        if header.nnz > total_elements {
            return Err(crate::error::BspcError::CorruptedData);
        }

        Ok(header)
    }
}

impl Default for BspcHeader {
    fn default() -> Self {
        Self::new()
    }
}

impl BspcHeader {
    /// Convert header to bytes for writing to file
    #[cfg(feature = "alloc")]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::SIZE);

        bytes.extend_from_slice(&self.magic);
        bytes.push(self.version);
        bytes.push(self.format_type);
        bytes.push(self.data_type);
        bytes.push(self.structure_flags);

        bytes.extend_from_slice(&self.nrows.to_le_bytes());
        bytes.extend_from_slice(&self.ncols.to_le_bytes());
        bytes.extend_from_slice(&self.nnz.to_le_bytes());
        bytes.extend_from_slice(&self.values_offset.to_le_bytes());
        bytes.extend_from_slice(&self.values_size.to_le_bytes());
        bytes.extend_from_slice(&self.indices_0_offset.to_le_bytes());
        bytes.extend_from_slice(&self.indices_0_size.to_le_bytes());
        bytes.extend_from_slice(&self.indices_1_offset.to_le_bytes());
        bytes.extend_from_slice(&self.indices_1_size.to_le_bytes());
        bytes.extend_from_slice(&self.pointers_offset.to_le_bytes());
        bytes.extend_from_slice(&self.pointers_size.to_le_bytes());
        bytes.extend_from_slice(&self.metadata_offset.to_le_bytes());
        bytes.extend_from_slice(&self.metadata_size.to_le_bytes());
        bytes.extend_from_slice(&self.bloom_filter_offset.to_le_bytes());
        bytes.extend_from_slice(&self.bloom_filter_size.to_le_bytes());
        bytes.extend_from_slice(&self.reserved);

        bytes
    }
}

/// Matrix format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixFormat {
    /// Coordinate format (COO)
    Coo = 0,
    /// Compressed Sparse Row (CSR)
    Csr = 1,
    /// Compressed Sparse Column (CSC)
    Csc = 2,
}

impl From<u8> for MatrixFormat {
    fn from(value: u8) -> Self {
        match value {
            0 => MatrixFormat::Coo,
            1 => MatrixFormat::Csr,
            2 => MatrixFormat::Csc,
            _ => MatrixFormat::Coo, // Default fallback
        }
    }
}

impl core::fmt::Display for MatrixFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MatrixFormat::Coo => write!(f, "COO"),
            MatrixFormat::Csr => write!(f, "CSR"),
            MatrixFormat::Csc => write!(f, "CSC"),
        }
    }
}

/// Data types supported in .bspc files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
    U32 = 4,
    U64 = 5,
}

impl From<u8> for DataType {
    fn from(value: u8) -> Self {
        match value {
            0 => DataType::F32,
            1 => DataType::F64,
            2 => DataType::I32,
            3 => DataType::I64,
            4 => DataType::U32,
            5 => DataType::U64,
            _ => DataType::F64, // Default fallback
        }
    }
}

impl core::fmt::Display for DataType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DataType::F32 => write!(f, "f32"),
            DataType::F64 => write!(f, "f64"),
            DataType::I32 => write!(f, "i32"),
            DataType::I64 => write!(f, "i64"),
            DataType::U32 => write!(f, "u32"),
            DataType::U64 => write!(f, "u64"),
        }
    }
}

impl DataType {
    /// Get the size in bytes for this data type
    pub const fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::U64 => 8,
        }
    }
}

/// Structure flags for matrix properties
pub mod structure_flags {
    /// Matrix is symmetric
    pub const SYMMETRIC: u8 = 1 << 0;
    /// Matrix is upper triangular
    pub const UPPER_TRIANGULAR: u8 = 1 << 1;
    /// Matrix is lower triangular
    pub const LOWER_TRIANGULAR: u8 = 1 << 2;
    /// Matrix has sorted indices
    pub const SORTED_INDICES: u8 = 1 << 3;
}
