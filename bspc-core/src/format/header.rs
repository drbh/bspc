//! Core BSPC header format definitions
//!
//! This module contains the main BSPC file header structure and related enums.

use core::mem::size_of;

#[cfg(feature = "alloc")]
extern crate alloc;

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
    /// Offset to values array from file start
    pub values_offset: u64,
    /// Size of values array in bytes
    pub values_size: u64,
    /// Offset to first index array (rows for COO, row indices for CSC)
    pub indices_0_offset: u64,
    /// Size of first index array in bytes
    pub indices_0_size: u64,
    /// Offset to second index array (cols for COO, col indices for CSR)
    pub indices_1_offset: u64,
    /// Size of second index array in bytes
    pub indices_1_size: u64,
    /// Offset to pointers array (for CSR/CSC formats)
    pub pointers_offset: u64,
    /// Size of pointers array in bytes
    pub pointers_size: u64,
    /// Offset to metadata region
    pub metadata_offset: u64,
    /// Size of metadata region in bytes
    pub metadata_size: u64,
    /// Offset to chunk bloom filter data
    pub bloom_filter_offset: u64,
    /// Size of chunk bloom filter data in bytes
    pub bloom_filter_size: u64,
    /// Reserved space for future extensions
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

    /// Set metadata region offset and size
    pub fn set_metadata_region(&mut self, offset: u64, size: u64) {
        self.metadata_offset = offset;
        self.metadata_size = size;
    }

    /// Set chunk bloom filter region offset and size
    pub fn set_chunk_bloom_filter_region(&mut self, offset: u64, size: u64) {
        self.bloom_filter_offset = offset;
        self.bloom_filter_size = size;
    }

    /// Validate the header structure
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version <= Self::VERSION
    }

    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::BspcError> {
        if bytes.len() < Self::SIZE {
            return Err(crate::BspcError::InsufficientBuffer);
        }

        // Validate magic bytes
        if bytes[0..4] != Self::MAGIC {
            return Err(crate::BspcError::InvalidHeader);
        }

        // Parse header fields (assuming little-endian)
        let version = bytes[4];
        let format_type = bytes[5];
        let data_type = bytes[6];
        let structure_flags = bytes[7];

        // Parse u64 fields
        let nrows = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let ncols = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let nnz = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);

        // Parse remaining u64 fields
        let values_offset = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
        ]);
        let values_size = u64::from_le_bytes([
            bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
        ]);
        let indices_0_offset = u64::from_le_bytes([
            bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
        ]);
        let indices_0_size = u64::from_le_bytes([
            bytes[56], bytes[57], bytes[58], bytes[59], bytes[60], bytes[61], bytes[62], bytes[63],
        ]);
        let indices_1_offset = u64::from_le_bytes([
            bytes[64], bytes[65], bytes[66], bytes[67], bytes[68], bytes[69], bytes[70], bytes[71],
        ]);
        let indices_1_size = u64::from_le_bytes([
            bytes[72], bytes[73], bytes[74], bytes[75], bytes[76], bytes[77], bytes[78], bytes[79],
        ]);
        let pointers_offset = u64::from_le_bytes([
            bytes[80], bytes[81], bytes[82], bytes[83], bytes[84], bytes[85], bytes[86], bytes[87],
        ]);
        let pointers_size = u64::from_le_bytes([
            bytes[88], bytes[89], bytes[90], bytes[91], bytes[92], bytes[93], bytes[94], bytes[95],
        ]);
        let metadata_offset = u64::from_le_bytes([
            bytes[96], bytes[97], bytes[98], bytes[99], bytes[100], bytes[101], bytes[102],
            bytes[103],
        ]);
        let metadata_size = u64::from_le_bytes([
            bytes[104], bytes[105], bytes[106], bytes[107], bytes[108], bytes[109], bytes[110],
            bytes[111],
        ]);
        let bloom_filter_offset = u64::from_le_bytes([
            bytes[112], bytes[113], bytes[114], bytes[115], bytes[116], bytes[117], bytes[118],
            bytes[119],
        ]);
        let bloom_filter_size = u64::from_le_bytes([
            bytes[120], bytes[121], bytes[122], bytes[123], bytes[124], bytes[125], bytes[126],
            bytes[127],
        ]);

        let mut reserved = [0u8; 32];
        reserved.copy_from_slice(&bytes[128..160]);

        Ok(Self {
            magic: Self::MAGIC,
            version,
            format_type,
            data_type,
            structure_flags,
            nrows,
            ncols,
            nnz,
            values_offset,
            values_size,
            indices_0_offset,
            indices_0_size,
            indices_1_offset,
            indices_1_size,
            pointers_offset,
            pointers_size,
            metadata_offset,
            metadata_size,
            bloom_filter_offset,
            bloom_filter_size,
            reserved,
        })
    }

    /// Convert header to bytes (requires alloc feature)
    #[cfg(feature = "alloc")]
    pub fn to_bytes(&self) -> alloc::vec::Vec<u8> {
        let mut bytes = alloc::vec::Vec::with_capacity(Self::SIZE);

        // Magic bytes and basic fields
        bytes.extend_from_slice(&self.magic);
        bytes.push(self.version);
        bytes.push(self.format_type);
        bytes.push(self.data_type);
        bytes.push(self.structure_flags);

        // u64 fields in little-endian
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

    /// Convert header to bytes array (no-std compatible)
    pub const fn to_bytes_array(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];

        // Magic bytes
        bytes[0] = self.magic[0];
        bytes[1] = self.magic[1];
        bytes[2] = self.magic[2];
        bytes[3] = self.magic[3];

        // Basic fields
        bytes[4] = self.version;
        bytes[5] = self.format_type;
        bytes[6] = self.data_type;
        bytes[7] = self.structure_flags;

        // u64 fields in little-endian
        let nrows_bytes = self.nrows.to_le_bytes();
        bytes[8] = nrows_bytes[0];
        bytes[9] = nrows_bytes[1];
        bytes[10] = nrows_bytes[2];
        bytes[11] = nrows_bytes[3];
        bytes[12] = nrows_bytes[4];
        bytes[13] = nrows_bytes[5];
        bytes[14] = nrows_bytes[6];
        bytes[15] = nrows_bytes[7];

        let ncols_bytes = self.ncols.to_le_bytes();
        bytes[16] = ncols_bytes[0];
        bytes[17] = ncols_bytes[1];
        bytes[18] = ncols_bytes[2];
        bytes[19] = ncols_bytes[3];
        bytes[20] = ncols_bytes[4];
        bytes[21] = ncols_bytes[5];
        bytes[22] = ncols_bytes[6];
        bytes[23] = ncols_bytes[7];

        let nnz_bytes = self.nnz.to_le_bytes();
        bytes[24] = nnz_bytes[0];
        bytes[25] = nnz_bytes[1];
        bytes[26] = nnz_bytes[2];
        bytes[27] = nnz_bytes[3];
        bytes[28] = nnz_bytes[4];
        bytes[29] = nnz_bytes[5];
        bytes[30] = nnz_bytes[6];
        bytes[31] = nnz_bytes[7];

        let values_offset_bytes = self.values_offset.to_le_bytes();
        bytes[32] = values_offset_bytes[0];
        bytes[33] = values_offset_bytes[1];
        bytes[34] = values_offset_bytes[2];
        bytes[35] = values_offset_bytes[3];
        bytes[36] = values_offset_bytes[4];
        bytes[37] = values_offset_bytes[5];
        bytes[38] = values_offset_bytes[6];
        bytes[39] = values_offset_bytes[7];

        let values_size_bytes = self.values_size.to_le_bytes();
        bytes[40] = values_size_bytes[0];
        bytes[41] = values_size_bytes[1];
        bytes[42] = values_size_bytes[2];
        bytes[43] = values_size_bytes[3];
        bytes[44] = values_size_bytes[4];
        bytes[45] = values_size_bytes[5];
        bytes[46] = values_size_bytes[6];
        bytes[47] = values_size_bytes[7];

        let indices_0_offset_bytes = self.indices_0_offset.to_le_bytes();
        bytes[48] = indices_0_offset_bytes[0];
        bytes[49] = indices_0_offset_bytes[1];
        bytes[50] = indices_0_offset_bytes[2];
        bytes[51] = indices_0_offset_bytes[3];
        bytes[52] = indices_0_offset_bytes[4];
        bytes[53] = indices_0_offset_bytes[5];
        bytes[54] = indices_0_offset_bytes[6];
        bytes[55] = indices_0_offset_bytes[7];

        let indices_0_size_bytes = self.indices_0_size.to_le_bytes();
        bytes[56] = indices_0_size_bytes[0];
        bytes[57] = indices_0_size_bytes[1];
        bytes[58] = indices_0_size_bytes[2];
        bytes[59] = indices_0_size_bytes[3];
        bytes[60] = indices_0_size_bytes[4];
        bytes[61] = indices_0_size_bytes[5];
        bytes[62] = indices_0_size_bytes[6];
        bytes[63] = indices_0_size_bytes[7];

        let indices_1_offset_bytes = self.indices_1_offset.to_le_bytes();
        bytes[64] = indices_1_offset_bytes[0];
        bytes[65] = indices_1_offset_bytes[1];
        bytes[66] = indices_1_offset_bytes[2];
        bytes[67] = indices_1_offset_bytes[3];
        bytes[68] = indices_1_offset_bytes[4];
        bytes[69] = indices_1_offset_bytes[5];
        bytes[70] = indices_1_offset_bytes[6];
        bytes[71] = indices_1_offset_bytes[7];

        let indices_1_size_bytes = self.indices_1_size.to_le_bytes();
        bytes[72] = indices_1_size_bytes[0];
        bytes[73] = indices_1_size_bytes[1];
        bytes[74] = indices_1_size_bytes[2];
        bytes[75] = indices_1_size_bytes[3];
        bytes[76] = indices_1_size_bytes[4];
        bytes[77] = indices_1_size_bytes[5];
        bytes[78] = indices_1_size_bytes[6];
        bytes[79] = indices_1_size_bytes[7];

        let pointers_offset_bytes = self.pointers_offset.to_le_bytes();
        bytes[80] = pointers_offset_bytes[0];
        bytes[81] = pointers_offset_bytes[1];
        bytes[82] = pointers_offset_bytes[2];
        bytes[83] = pointers_offset_bytes[3];
        bytes[84] = pointers_offset_bytes[4];
        bytes[85] = pointers_offset_bytes[5];
        bytes[86] = pointers_offset_bytes[6];
        bytes[87] = pointers_offset_bytes[7];

        let pointers_size_bytes = self.pointers_size.to_le_bytes();
        bytes[88] = pointers_size_bytes[0];
        bytes[89] = pointers_size_bytes[1];
        bytes[90] = pointers_size_bytes[2];
        bytes[91] = pointers_size_bytes[3];
        bytes[92] = pointers_size_bytes[4];
        bytes[93] = pointers_size_bytes[5];
        bytes[94] = pointers_size_bytes[6];
        bytes[95] = pointers_size_bytes[7];

        let metadata_offset_bytes = self.metadata_offset.to_le_bytes();
        bytes[96] = metadata_offset_bytes[0];
        bytes[97] = metadata_offset_bytes[1];
        bytes[98] = metadata_offset_bytes[2];
        bytes[99] = metadata_offset_bytes[3];
        bytes[100] = metadata_offset_bytes[4];
        bytes[101] = metadata_offset_bytes[5];
        bytes[102] = metadata_offset_bytes[6];
        bytes[103] = metadata_offset_bytes[7];

        let metadata_size_bytes = self.metadata_size.to_le_bytes();
        bytes[104] = metadata_size_bytes[0];
        bytes[105] = metadata_size_bytes[1];
        bytes[106] = metadata_size_bytes[2];
        bytes[107] = metadata_size_bytes[3];
        bytes[108] = metadata_size_bytes[4];
        bytes[109] = metadata_size_bytes[5];
        bytes[110] = metadata_size_bytes[6];
        bytes[111] = metadata_size_bytes[7];

        let bloom_filter_offset_bytes = self.bloom_filter_offset.to_le_bytes();
        bytes[112] = bloom_filter_offset_bytes[0];
        bytes[113] = bloom_filter_offset_bytes[1];
        bytes[114] = bloom_filter_offset_bytes[2];
        bytes[115] = bloom_filter_offset_bytes[3];
        bytes[116] = bloom_filter_offset_bytes[4];
        bytes[117] = bloom_filter_offset_bytes[5];
        bytes[118] = bloom_filter_offset_bytes[6];
        bytes[119] = bloom_filter_offset_bytes[7];

        let bloom_filter_size_bytes = self.bloom_filter_size.to_le_bytes();
        bytes[120] = bloom_filter_size_bytes[0];
        bytes[121] = bloom_filter_size_bytes[1];
        bytes[122] = bloom_filter_size_bytes[2];
        bytes[123] = bloom_filter_size_bytes[3];
        bytes[124] = bloom_filter_size_bytes[4];
        bytes[125] = bloom_filter_size_bytes[5];
        bytes[126] = bloom_filter_size_bytes[6];
        bytes[127] = bloom_filter_size_bytes[7];

        // Reserved bytes (already initialized to 0)
        let mut i = 0;
        while i < 32 {
            bytes[128 + i] = self.reserved[i];
            i += 1;
        }

        bytes
    }
}

impl Default for BspcHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix storage formats supported by BSPC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixFormat {
    /// Coordinate (COO) format - row, col, value triplets
    Coo = 0,
    /// Compressed Sparse Row (CSR) format
    Csr = 1,
    /// Compressed Sparse Column (CSC) format
    Csc = 2,
}

impl MatrixFormat {
    /// Convert from u8 representation
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(MatrixFormat::Coo),
            1 => Some(MatrixFormat::Csr),
            2 => Some(MatrixFormat::Csc),
            _ => None,
        }
    }

    /// Convert to u8 representation
    pub const fn to_u8(self) -> u8 {
        self as u8
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

/// Data types supported by BSPC format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    /// 32-bit floating point
    F32 = 0,
    /// 64-bit floating point
    F64 = 1,
    /// 32-bit signed integer
    I32 = 2,
    /// 64-bit signed integer
    I64 = 3,
    /// 32-bit unsigned integer
    U32 = 4,
    /// 64-bit unsigned integer
    U64 = 5,
}

impl DataType {
    /// Convert from u8 representation
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(DataType::F32),
            1 => Some(DataType::F64),
            2 => Some(DataType::I32),
            3 => Some(DataType::I64),
            4 => Some(DataType::U32),
            5 => Some(DataType::U64),
            _ => None,
        }
    }

    /// Convert to u8 representation
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Get the size in bytes for this data type
    pub const fn size_bytes(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::U64 => 8,
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
