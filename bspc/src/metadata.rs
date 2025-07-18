//! Structured metadata support for BSPC files
//!
//! This module provides structured access to metadata stored in BSPC files,
//! including fast O(1) label lookups for row and column labels.

use binsparse_rs::{Error, Result};
use std::convert::TryInto;

/// Fixed-size metadata header (40 bytes, 8-byte aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
    /// Magic bytes for metadata
    pub const MAGIC: [u8; 4] = *b"META";

    /// Current metadata version
    pub const VERSION: u8 = 1;

    /// Size of the metadata header in bytes
    pub const SIZE: usize = 40;

    /// Create a new metadata header with default values
    pub const fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            _padding: [0; 3],
            row_labels_offset: 0,
            row_labels_size: 0,
            col_labels_offset: 0,
            col_labels_size: 0,
        }
    }

    /// Validate the metadata header
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version <= Self::VERSION
    }

    /// Parse metadata header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::InvalidState("Metadata header too small"));
        }

        // Validate magic bytes
        if bytes[0..4] != Self::MAGIC {
            return Err(Error::InvalidState("Invalid metadata magic bytes"));
        }

        let version = bytes[4];
        if version > Self::VERSION {
            return Err(Error::InvalidState("Unsupported metadata version"));
        }

        // Parse fields
        let row_labels_offset = u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid row labels offset"))?,
        );
        let row_labels_size = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid row labels size"))?,
        );
        let col_labels_offset = u64::from_le_bytes(
            bytes[24..32]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid column labels offset"))?,
        );
        let col_labels_size = u64::from_le_bytes(
            bytes[32..40]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid column labels size"))?,
        );

        Ok(Self {
            magic: Self::MAGIC,
            version,
            _padding: [0; 3],
            row_labels_offset,
            row_labels_size,
            col_labels_offset,
            col_labels_size,
        })
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        // padding bytes remain zero
        bytes[8..16].copy_from_slice(&self.row_labels_offset.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.row_labels_size.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.col_labels_offset.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.col_labels_size.to_le_bytes());
        bytes
    }
}

/// Memory layout for fast row/column label lookup
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LabelArray {
    /// Number of labels
    pub count: u32,
    /// Bytes per label (fixed width)
    pub stride: u32,
    // Followed by: data: [u8; count * stride]
}

impl LabelArray {
    /// Size of the label array header in bytes
    pub const HEADER_SIZE: usize = 8;

    /// Parse label array header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(Error::InvalidState("Label array header too small"));
        }

        let count = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid label count"))?,
        );
        let stride = u32::from_le_bytes(
            bytes[4..8]
                .try_into()
                .map_err(|_| Error::InvalidState("Invalid label stride"))?,
        );

        // Validate stride is reasonable (1-64KB per label)
        if stride == 0 || stride > 65536 {
            return Err(Error::InvalidState("Invalid label stride"));
        }

        Ok(Self { count, stride })
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> [u8; Self::HEADER_SIZE] {
        let mut bytes = [0u8; Self::HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.stride.to_le_bytes());
        bytes
    }

    /// Calculate total size including data
    pub fn total_size(&self) -> Result<usize> {
        let data_size = (self.count as usize)
            .checked_mul(self.stride as usize)
            .ok_or(Error::InvalidState("Label array size overflow"))?;

        Self::HEADER_SIZE
            .checked_add(data_size)
            .ok_or(Error::InvalidState("Label array total size overflow"))
    }
}

/// View for accessing structured metadata with fast lookups
pub struct MetadataView<'a> {
    data: &'a [u8],
    header: BspcMetadataHeader,
}

impl<'a> MetadataView<'a> {
    /// Create a new metadata view from raw bytes
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let header = BspcMetadataHeader::from_bytes(data)?;

        // Validate offsets are within bounds
        if header.row_labels_offset as usize + header.row_labels_size as usize > data.len() {
            return Err(Error::InvalidState("Row labels extend beyond metadata"));
        }
        if header.col_labels_offset as usize + header.col_labels_size as usize > data.len() {
            return Err(Error::InvalidState("Column labels extend beyond metadata"));
        }

        Ok(Self { data, header })
    }

    /// Get the metadata header
    pub fn header(&self) -> &BspcMetadataHeader {
        &self.header
    }

    /// Get row labels array header
    pub fn row_labels_array(&self) -> Result<LabelArray> {
        if self.header.row_labels_size == 0 {
            return Err(Error::InvalidState("No row labels present"));
        }

        let start = self.header.row_labels_offset as usize;
        let end = start + self.header.row_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Row labels extend beyond data"));
        }

        let labels_data = &self.data[start..end];
        LabelArray::from_bytes(labels_data)
    }

    /// Get column labels array header
    pub fn col_labels_array(&self) -> Result<LabelArray> {
        if self.header.col_labels_size == 0 {
            return Err(Error::InvalidState("No column labels present"));
        }

        let start = self.header.col_labels_offset as usize;
        let end = start + self.header.col_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Column labels extend beyond data"));
        }

        let labels_data = &self.data[start..end];
        LabelArray::from_bytes(labels_data)
    }

    /// Get O(1) access to row label by index
    pub fn row_label(&self, row_idx: u32) -> Result<&[u8]> {
        let labels = self.row_labels_array()?;

        if row_idx >= labels.count {
            return Err(Error::InvalidState("Row index out of bounds"));
        }

        let start = self.header.row_labels_offset as usize
            + LabelArray::HEADER_SIZE
            + (row_idx as usize * labels.stride as usize);
        let end = start + labels.stride as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Row label extends beyond data"));
        }

        Ok(&self.data[start..end])
    }

    /// Get O(1) access to column label by index
    pub fn col_label(&self, col_idx: u32) -> Result<&[u8]> {
        let labels = self.col_labels_array()?;

        if col_idx >= labels.count {
            return Err(Error::InvalidState("Column index out of bounds"));
        }

        let start = self.header.col_labels_offset as usize
            + LabelArray::HEADER_SIZE
            + (col_idx as usize * labels.stride as usize);
        let end = start + labels.stride as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Column label extends beyond data"));
        }

        Ok(&self.data[start..end])
    }

    /// Get memory-mapped slice access to all row labels (no allocations)
    pub fn row_labels_slice(&self) -> Result<&[u8]> {
        if self.header.row_labels_size == 0 {
            return Err(Error::InvalidState("No row labels present"));
        }

        let start = self.header.row_labels_offset as usize;
        let end = start + self.header.row_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Row labels extend beyond data"));
        }

        Ok(&self.data[start..end])
    }

    /// Get memory-mapped slice access to all column labels (no allocations)
    pub fn col_labels_slice(&self) -> Result<&[u8]> {
        if self.header.col_labels_size == 0 {
            return Err(Error::InvalidState("No column labels present"));
        }

        let start = self.header.col_labels_offset as usize;
        let end = start + self.header.col_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Column labels extend beyond data"));
        }

        Ok(&self.data[start..end])
    }
}

/// Builder for creating structured metadata
pub struct MetadataBuilder {
    row_labels: Vec<Vec<u8>>,
    col_labels: Vec<Vec<u8>>,
    label_stride: u32,
}

impl MetadataBuilder {
    /// Create a new metadata builder
    pub fn new(label_stride: u32) -> Self {
        Self {
            row_labels: Vec::new(),
            col_labels: Vec::new(),
            label_stride,
        }
    }

    /// Add row labels
    pub fn with_row_labels(mut self, labels: Vec<Vec<u8>>) -> Result<Self> {
        // Validate all labels fit in stride
        for label in &labels {
            if label.len() > self.label_stride as usize {
                return Err(Error::InvalidState("Row label exceeds stride"));
            }
        }
        self.row_labels = labels;
        Ok(self)
    }

    /// Add column labels
    pub fn with_col_labels(mut self, labels: Vec<Vec<u8>>) -> Result<Self> {
        // Validate all labels fit in stride
        for label in &labels {
            if label.len() > self.label_stride as usize {
                return Err(Error::InvalidState("Column label exceeds stride"));
            }
        }
        self.col_labels = labels;
        Ok(self)
    }

    /// Build the metadata into a byte vector
    pub fn build(self) -> Result<Vec<u8>> {
        let mut metadata = Vec::new();

        // Calculate layout
        let header_size = BspcMetadataHeader::SIZE;
        let row_data_start = header_size;
        let row_data_size = if self.row_labels.is_empty() {
            0
        } else {
            LabelArray::HEADER_SIZE + (self.row_labels.len() * self.label_stride as usize)
        };
        let col_data_start = row_data_start + row_data_size;
        let col_data_size = if self.col_labels.is_empty() {
            0
        } else {
            LabelArray::HEADER_SIZE + (self.col_labels.len() * self.label_stride as usize)
        };

        // Create header
        let mut header = BspcMetadataHeader::new();
        if !self.row_labels.is_empty() {
            header.row_labels_offset = row_data_start as u64;
            header.row_labels_size = row_data_size as u64;
        }
        if !self.col_labels.is_empty() {
            header.col_labels_offset = col_data_start as u64;
            header.col_labels_size = col_data_size as u64;
        }

        // Write header
        metadata.extend_from_slice(&header.to_bytes());

        // Write row labels
        if !self.row_labels.is_empty() {
            let row_array = LabelArray {
                count: self.row_labels.len() as u32,
                stride: self.label_stride,
            };
            metadata.extend_from_slice(&row_array.to_bytes());

            for label in &self.row_labels {
                let mut padded_label = vec![0u8; self.label_stride as usize];
                padded_label[..label.len()].copy_from_slice(label);
                metadata.extend_from_slice(&padded_label);
            }
        }

        // Write column labels
        if !self.col_labels.is_empty() {
            let col_array = LabelArray {
                count: self.col_labels.len() as u32,
                stride: self.label_stride,
            };
            metadata.extend_from_slice(&col_array.to_bytes());

            for label in &self.col_labels {
                let mut padded_label = vec![0u8; self.label_stride as usize];
                padded_label[..label.len()].copy_from_slice(label);
                metadata.extend_from_slice(&padded_label);
            }
        }

        Ok(metadata)
    }
}

/// Utility function to align to 8-byte boundary
pub fn align_to_8(offset: usize) -> usize {
    (offset + 7) & !7
}
