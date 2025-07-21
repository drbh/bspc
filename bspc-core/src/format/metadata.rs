//! Metadata format definitions for BSPC specification
//!
//! This module defines the binary layout for metadata sections in BSPC files.
//! Contains pure format definitions with validation - no I/O operations.

use crate::{BspcError, Result};
use super::constants::metadata::*;

/// Fixed-size metadata header (40 bytes, 8-byte aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BspcMetadataHeader {
    /// Magic bytes: "META"
    pub magic: [u8; 4],
    /// Version number (1)
    pub version: u8,
    /// Padding for alignment
    pub _padding: [u8; 3],
    /// Offset to row labels array from metadata start
    pub row_labels_offset: u64,
    /// Size of row labels array in bytes
    pub row_labels_size: u64,
    /// Offset to column labels array from metadata start
    pub col_labels_offset: u64,
    /// Size of column labels array in bytes
    pub col_labels_size: u64,
}

impl Default for BspcMetadataHeader {
    fn default() -> Self {
        Self::new()
    }
}

impl BspcMetadataHeader {
    /// Create a new metadata header with default values
    pub const fn new() -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            _padding: [0; 3],
            row_labels_offset: 0,
            row_labels_size: 0,
            col_labels_offset: 0,
            col_labels_size: 0,
        }
    }

    /// Validate the metadata header
    pub const fn is_valid(&self) -> bool {
        // Check magic bytes
        let magic_valid = self.magic[0] == MAGIC[0] 
            && self.magic[1] == MAGIC[1]
            && self.magic[2] == MAGIC[2] 
            && self.magic[3] == MAGIC[3];
        
        // Check version
        let version_valid = self.version <= VERSION;
        
        magic_valid && version_valid
    }

    /// Parse metadata header from bytes
    pub const fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(BspcError::InsufficientBuffer);
        }

        // Validate magic bytes
        if bytes[0] != MAGIC[0] || bytes[1] != MAGIC[1] 
            || bytes[2] != MAGIC[2] || bytes[3] != MAGIC[3] {
            return Err(BspcError::InvalidMetadata);
        }

        let version = bytes[4];
        if version > VERSION {
            return Err(BspcError::UnsupportedFormat);
        }

        // Parse fields using const-friendly approach
        let row_labels_offset = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let row_labels_size = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let col_labels_offset = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let col_labels_size = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35],
            bytes[36], bytes[37], bytes[38], bytes[39],
        ]);

        Ok(Self {
            magic: MAGIC,
            version,
            _padding: [0; 3],
            row_labels_offset,
            row_labels_size,
            col_labels_offset,
            col_labels_size,
        })
    }

    /// Convert header to bytes (const-friendly)
    pub const fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        
        // Magic bytes
        bytes[0] = self.magic[0];
        bytes[1] = self.magic[1];
        bytes[2] = self.magic[2];
        bytes[3] = self.magic[3];
        
        // Version
        bytes[4] = self.version;
        // Padding bytes 5-7 already zeroed
        
        // Row labels offset
        let row_offset_bytes = self.row_labels_offset.to_le_bytes();
        bytes[8] = row_offset_bytes[0];
        bytes[9] = row_offset_bytes[1];
        bytes[10] = row_offset_bytes[2];
        bytes[11] = row_offset_bytes[3];
        bytes[12] = row_offset_bytes[4];
        bytes[13] = row_offset_bytes[5];
        bytes[14] = row_offset_bytes[6];
        bytes[15] = row_offset_bytes[7];
        
        // Row labels size
        let row_size_bytes = self.row_labels_size.to_le_bytes();
        bytes[16] = row_size_bytes[0];
        bytes[17] = row_size_bytes[1];
        bytes[18] = row_size_bytes[2];
        bytes[19] = row_size_bytes[3];
        bytes[20] = row_size_bytes[4];
        bytes[21] = row_size_bytes[5];
        bytes[22] = row_size_bytes[6];
        bytes[23] = row_size_bytes[7];
        
        // Column labels offset
        let col_offset_bytes = self.col_labels_offset.to_le_bytes();
        bytes[24] = col_offset_bytes[0];
        bytes[25] = col_offset_bytes[1];
        bytes[26] = col_offset_bytes[2];
        bytes[27] = col_offset_bytes[3];
        bytes[28] = col_offset_bytes[4];
        bytes[29] = col_offset_bytes[5];
        bytes[30] = col_offset_bytes[6];
        bytes[31] = col_offset_bytes[7];
        
        // Column labels size
        let col_size_bytes = self.col_labels_size.to_le_bytes();
        bytes[32] = col_size_bytes[0];
        bytes[33] = col_size_bytes[1];
        bytes[34] = col_size_bytes[2];
        bytes[35] = col_size_bytes[3];
        bytes[36] = col_size_bytes[4];
        bytes[37] = col_size_bytes[5];
        bytes[38] = col_size_bytes[6];
        bytes[39] = col_size_bytes[7];
        
        bytes
    }
}

/// Label array header (8 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelArrayHeader {
    /// Number of labels in the array
    pub count: u32,
    /// Fixed stride (width) of each label in bytes
    pub stride: u32,
}

impl LabelArrayHeader {
    /// Create a new label array header
    pub const fn new(count: u32, stride: u32) -> Self {
        Self { count, stride }
    }

    /// Parse from bytes
    pub const fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < LABEL_ARRAY_HEADER_SIZE {
            return Err(BspcError::InsufficientBuffer);
        }

        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let stride = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        // Validate stride
        if stride == 0 || stride > crate::format::constants::MAX_LABEL_STRIDE {
            return Err(BspcError::InvalidMetadata);
        }

        Ok(Self { count, stride })
    }

    /// Convert to bytes
    pub const fn to_bytes(&self) -> [u8; LABEL_ARRAY_HEADER_SIZE] {
        let mut bytes = [0u8; LABEL_ARRAY_HEADER_SIZE];
        
        let count_bytes = self.count.to_le_bytes();
        bytes[0] = count_bytes[0];
        bytes[1] = count_bytes[1];
        bytes[2] = count_bytes[2];
        bytes[3] = count_bytes[3];
        
        let stride_bytes = self.stride.to_le_bytes();
        bytes[4] = stride_bytes[0];
        bytes[5] = stride_bytes[1];
        bytes[6] = stride_bytes[2];
        bytes[7] = stride_bytes[3];
        
        bytes
    }

    /// Calculate total size of the label data
    pub const fn total_size(&self) -> usize {
        self.count as usize * self.stride as usize
    }
}