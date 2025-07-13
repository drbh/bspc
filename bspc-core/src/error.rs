//! Error types for BSPC operations

/// Errors that can occur during BSPC operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BspcError {
    /// Invalid header format
    InvalidHeader,
    /// Index out of bounds
    IndexOutOfBounds,
    /// Unsupported format version
    UnsupportedFormat,
    /// Invalid chunk metadata
    InvalidChunk,
    /// Data corruption detected
    CorruptedData,
    /// Insufficient buffer space
    InsufficientBuffer,
}

impl core::fmt::Display for BspcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            BspcError::InvalidHeader => "Invalid BSPC header",
            BspcError::IndexOutOfBounds => "Index out of bounds",
            BspcError::UnsupportedFormat => "Unsupported format version",
            BspcError::InvalidChunk => "Invalid chunk metadata",
            BspcError::CorruptedData => "Data corruption detected",
            BspcError::InsufficientBuffer => "Insufficient buffer space",
        };
        write!(f, "{msg}")
    }
}

/// Result type for BSPC operations
pub type Result<T> = core::result::Result<T, BspcError>;
