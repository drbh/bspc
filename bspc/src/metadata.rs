//! Structured metadata support for BSPC files
//!
//! This module provides structured access to metadata stored in BSPC files,
//! including fast O(1) label lookups for row and column labels.

use binsparse_rs::{Error, Result};

// Re-export metadata format definitions from bspc-core
pub use bspc_core::format::metadata::{BspcMetadataHeader, LabelArrayHeader};

/// Fast label lookup array for matrix rows/columns
pub struct LabelArray<'a> {
    data: &'a [u8],
    header: LabelArrayHeader,
}

impl<'a> LabelArray<'a> {
    /// Create from bytes with validation
    pub fn from_bytes(data: &'a [u8]) -> Result<Self> {
        let header = LabelArrayHeader::from_bytes(data)
            .map_err(|_| Error::InvalidState("Invalid label array header"))?;
        Ok(Self { data, header })
    }

    /// Get label by index
    pub fn get_label(&self, index: u32) -> Result<&[u8]> {
        if index >= self.header.count {
            return Err(Error::InvalidState("Label index out of bounds"));
        }

        let start = bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE
            + (index as usize * self.header.stride as usize);
        let end = start + self.header.stride as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Label extends beyond data"));
        }

        Ok(&self.data[start..end])
    }

    /// Get label count
    pub fn count(&self) -> u32 {
        self.header.count
    }

    /// Get label stride (size of each label in bytes)
    pub fn stride(&self) -> u32 {
        self.header.stride
    }
}

/// High-level metadata view with efficient label access
pub struct MetadataView<'a> {
    data: &'a [u8],
    header: BspcMetadataHeader,
}

impl<'a> MetadataView<'a> {
    /// Create from metadata bytes
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let header = BspcMetadataHeader::from_bytes(data)
            .map_err(|_| Error::InvalidState("Invalid metadata header"))?;
        Ok(Self { data, header })
    }

    /// Get row labels array if present
    pub fn row_labels(&self) -> Result<Option<LabelArray<'_>>> {
        if self.header.row_labels_size == 0 {
            return Ok(None);
        }

        let start = self.header.row_labels_offset as usize;
        let end = start + self.header.row_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Row labels extend beyond metadata"));
        }

        Ok(Some(LabelArray::from_bytes(&self.data[start..end])?))
    }

    /// Get column labels array if present  
    pub fn col_labels(&self) -> Result<Option<LabelArray<'_>>> {
        if self.header.col_labels_size == 0 {
            return Ok(None);
        }

        let start = self.header.col_labels_offset as usize;
        let end = start + self.header.col_labels_size as usize;

        if end > self.data.len() {
            return Err(Error::InvalidState("Column labels extend beyond metadata"));
        }

        Ok(Some(LabelArray::from_bytes(&self.data[start..end])?))
    }

    /// Get specific row label by index
    pub fn row_label(&self, index: u32) -> Result<Option<&[u8]>> {
        if self.header.row_labels_size == 0 {
            return Ok(None);
        }

        let labels_start = self.header.row_labels_offset as usize;
        let labels_end = labels_start + self.header.row_labels_size as usize;

        if labels_end > self.data.len() {
            return Err(Error::InvalidState("Row labels extend beyond metadata"));
        }

        let labels_data = &self.data[labels_start..labels_end];

        // Parse header manually to avoid lifetime issues
        if labels_data.len() < bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE {
            return Err(Error::InvalidState("Invalid label array header"));
        }

        let count = u32::from_le_bytes([
            labels_data[0],
            labels_data[1],
            labels_data[2],
            labels_data[3],
        ]);
        let stride = u32::from_le_bytes([
            labels_data[4],
            labels_data[5],
            labels_data[6],
            labels_data[7],
        ]);

        if index >= count {
            return Err(Error::InvalidState("Label index out of bounds"));
        }

        let label_start = bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE
            + (index as usize * stride as usize);
        let label_end = label_start + stride as usize;

        if label_end > labels_data.len() {
            return Err(Error::InvalidState("Label extends beyond data"));
        }

        Ok(Some(&labels_data[label_start..label_end]))
    }

    /// Get specific column label by index
    pub fn col_label(&self, index: u32) -> Result<Option<&[u8]>> {
        if self.header.col_labels_size == 0 {
            return Ok(None);
        }

        let labels_start = self.header.col_labels_offset as usize;
        let labels_end = labels_start + self.header.col_labels_size as usize;

        if labels_end > self.data.len() {
            return Err(Error::InvalidState("Column labels extend beyond metadata"));
        }

        let labels_data = &self.data[labels_start..labels_end];

        // Parse header manually to avoid lifetime issues
        if labels_data.len() < bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE {
            return Err(Error::InvalidState("Invalid label array header"));
        }

        let count = u32::from_le_bytes([
            labels_data[0],
            labels_data[1],
            labels_data[2],
            labels_data[3],
        ]);
        let stride = u32::from_le_bytes([
            labels_data[4],
            labels_data[5],
            labels_data[6],
            labels_data[7],
        ]);

        if index >= count {
            return Err(Error::InvalidState("Label index out of bounds"));
        }

        let label_start = bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE
            + (index as usize * stride as usize);
        let label_end = label_start + stride as usize;

        if label_end > labels_data.len() {
            return Err(Error::InvalidState("Label extends beyond data"));
        }

        Ok(Some(&labels_data[label_start..label_end]))
    }

    /// Get row labels array (compatibility method)
    pub fn row_labels_array(&self) -> Result<Option<LabelArray<'_>>> {
        self.row_labels()
    }

    /// Get column labels array (compatibility method)
    pub fn col_labels_array(&self) -> Result<Option<LabelArray<'_>>> {
        self.col_labels()
    }
}

/// Builder for creating metadata
pub struct MetadataBuilder {
    row_labels: Option<Vec<Vec<u8>>>,
    col_labels: Option<Vec<Vec<u8>>>,
}

impl MetadataBuilder {
    /// Create new metadata builder
    pub fn new() -> Self {
        Self {
            row_labels: None,
            col_labels: None,
        }
    }

    /// Add row labels
    pub fn with_row_labels(mut self, labels: Vec<Vec<u8>>) -> Self {
        self.row_labels = Some(labels);
        self
    }

    /// Add column labels
    pub fn with_col_labels(mut self, labels: Vec<Vec<u8>>) -> Self {
        self.col_labels = Some(labels);
        self
    }

    /// Build metadata bytes
    pub fn build(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Calculate offsets
        let header_size = bspc_core::format::constants::metadata::HEADER_SIZE;
        let mut current_offset = header_size as u64;

        let (row_labels_offset, row_labels_size) = if let Some(ref labels) = self.row_labels {
            let offset = current_offset;
            let size = self.calculate_label_array_size(labels)?;
            current_offset += size;
            (offset, size)
        } else {
            (0, 0)
        };

        let (col_labels_offset, col_labels_size) = if let Some(ref labels) = self.col_labels {
            let offset = current_offset;
            let size = self.calculate_label_array_size(labels)?;
            (offset, size)
        } else {
            (0, 0)
        };

        // Create header manually since there are no builder methods
        let mut header = BspcMetadataHeader::new();
        header.row_labels_offset = row_labels_offset;
        header.row_labels_size = row_labels_size;
        header.col_labels_offset = col_labels_offset;
        header.col_labels_size = col_labels_size;

        // Write header
        result.extend_from_slice(&header.to_bytes());

        // Write row labels if present
        if let Some(ref labels) = self.row_labels {
            self.write_label_array(&mut result, labels)?;
        }

        // Write column labels if present
        if let Some(ref labels) = self.col_labels {
            self.write_label_array(&mut result, labels)?;
        }

        Ok(result)
    }

    fn calculate_label_array_size(&self, labels: &[Vec<u8>]) -> Result<u64> {
        if labels.is_empty() {
            return Ok(0);
        }

        let max_len = labels.iter().map(|l| l.len()).max().unwrap_or(0);
        let stride = max_len.next_power_of_two().max(4); // At least 4 bytes, power of 2
        let total_size = bspc_core::format::constants::metadata::LABEL_ARRAY_HEADER_SIZE
            + (labels.len() * stride);

        Ok(total_size as u64)
    }

    fn write_label_array(&self, buf: &mut Vec<u8>, labels: &[Vec<u8>]) -> Result<()> {
        if labels.is_empty() {
            return Ok(());
        }

        let max_len = labels.iter().map(|l| l.len()).max().unwrap_or(0);
        let stride = max_len.next_power_of_two().max(4);

        // Write header
        let header = LabelArrayHeader::new(labels.len() as u32, stride as u32);
        buf.extend_from_slice(&header.to_bytes());

        // Write labels with padding
        for label in labels {
            buf.extend_from_slice(label);
            // Pad to stride
            buf.resize(buf.len() + (stride - label.len()), 0);
        }

        Ok(())
    }
}

impl Default for MetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function for 8-byte alignment
pub fn align_to_8(value: u64) -> u64 {
    (value + 7) & !7
}
