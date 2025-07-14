//! Binary sparse matrix format definitions
//!
//! This module defines the core data structures and binary format
//! specifications for .bspc files.

use core::mem::size_of;

/// Fixed-size header for .bspc files
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
    /// Number of rows
    pub nrows: u32,
    /// Number of columns
    pub ncols: u32,
    /// Number of non-zero elements
    pub nnz: u32,
    /// Offset to values array
    pub values_offset: u32,
    /// Size of values array in bytes
    pub values_size: u32,
    /// Offset to first index array
    pub indices_0_offset: u32,
    /// Size of first index array in bytes
    pub indices_0_size: u32,
    /// Offset to second index array (if applicable)
    pub indices_1_offset: u32,
    /// Size of second index array in bytes
    pub indices_1_size: u32,
    /// Offset to pointers array (for CSR/CSC)
    pub pointers_offset: u32,
    /// Size of pointers array in bytes
    pub pointers_size: u32,
    /// Reserved bytes for future use
    pub reserved: [u8; 16],
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
            reserved: [0; 16],
        }
    }

    /// Validate the header magic and version
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version <= Self::VERSION
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

        // Read fields one by one to avoid unsafe pointer casting
        let mut header = Self::new();
        header.magic.copy_from_slice(&bytes[0..4]);
        header.version = bytes[4];
        header.format_type = bytes[5];
        header.data_type = bytes[6];
        header.structure_flags = bytes[7];

        // Read u32 fields in little-endian format
        header.nrows = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        header.ncols = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        header.nnz = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        header.values_offset = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        header.values_size = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        header.indices_0_offset = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        header.indices_0_size = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        header.indices_1_offset = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        header.indices_1_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        header.pointers_offset = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        header.pointers_size = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]);

        // Copy reserved bytes
        header.reserved.copy_from_slice(&bytes[52..68]);

        // Validate the header structure
        if !header.is_valid() {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Additional validation
        if header.nrows == 0 || header.ncols == 0 {
            return Err(crate::error::BspcError::InvalidHeader);
        }

        // Check for potential integer overflow in matrix size
        let total_elements = (header.nrows as u64)
            .checked_mul(header.ncols as u64)
            .ok_or(crate::error::BspcError::InvalidHeader)?;

        if header.nnz as u64 > total_elements {
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
