//! Comprehensive error taxonomy for BSPC operations
//!
//! This module provides a Linux kernel style error classification system
//! with distinct error codes for different categories of failures.

/// Errors that can occur during BSPC operations
///
/// Error codes are organized by category with distinct numeric ranges
/// to enable efficient error handling and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BspcError {
    // Protocol errors (wire format issues) - 1-15
    /// Invalid BSPC header format or magic bytes
    InvalidHeader = 1,
    /// Invalid metadata section format
    InvalidMetadata = 2,
    /// Unsupported format version
    UnsupportedFormat = 3,
    /// Data corruption detected during parsing
    CorruptedData = 4,

    // Boundary errors (size/alignment issues) - 16-31
    /// Index out of bounds access
    IndexOutOfBounds = 16,
    /// Array size would overflow
    ArraySizeOverflow = 17,
    /// Array alignment requirements not met
    ArrayAlignment = 18,
    /// Insufficient buffer space for operation
    InsufficientBuffer = 19,

    // Semantic errors (logical consistency) - 32-47
    /// Invalid range specification
    InvalidRange = 32,
    /// Invalid label format or content
    InvalidLabel = 33,
    /// Invalid matrix element type
    InvalidElement = 34,
    /// Invalid chunk metadata or boundaries
    InvalidChunk = 35,
}

impl BspcError {
    /// Get the error category for this error
    pub const fn category(&self) -> ErrorCategory {
        match *self as u8 {
            1..=15 => ErrorCategory::Protocol,
            16..=31 => ErrorCategory::Boundary,
            32..=47 => ErrorCategory::Semantic,
            _ => ErrorCategory::Unknown,
        }
    }

    /// Get the numeric error code
    pub const fn code(&self) -> u8 {
        *self as u8
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Wire format and protocol errors
    Protocol,
    /// Size and alignment boundary errors  
    Boundary,
    /// Logical consistency and semantic errors
    Semantic,
    /// Unknown/undefined category
    Unknown,
}

impl core::fmt::Display for BspcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            // Protocol errors
            BspcError::InvalidHeader => "Invalid BSPC header format or magic bytes",
            BspcError::InvalidMetadata => "Invalid metadata section format",
            BspcError::UnsupportedFormat => "Unsupported format version",
            BspcError::CorruptedData => "Data corruption detected during parsing",

            // Boundary errors
            BspcError::IndexOutOfBounds => "Index out of bounds access",
            BspcError::ArraySizeOverflow => "Array size would overflow",
            BspcError::ArrayAlignment => "Array alignment requirements not met",
            BspcError::InsufficientBuffer => "Insufficient buffer space for operation",

            // Semantic errors
            BspcError::InvalidRange => "Invalid range specification",
            BspcError::InvalidLabel => "Invalid label format or content",
            BspcError::InvalidElement => "Invalid matrix element type",
            BspcError::InvalidChunk => "Invalid chunk metadata or boundaries",
        };
        write!(f, "{msg}")
    }
}

/// Result type for BSPC operations
pub type Result<T> = core::result::Result<T, BspcError>;
