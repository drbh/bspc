//! Format constants and magic bytes for BSPC specification

/// Default alignment boundary for all data structures
pub const ALIGNMENT_BOUNDARY: usize = 8;

/// Maximum reasonable label stride (64KB)
pub const MAX_LABEL_STRIDE: u32 = 65536;

/// Maximum chunk count to prevent memory exhaustion
pub const MAX_CHUNK_COUNT: usize = 1_000_000;

/// Metadata format constants
pub mod metadata {
    /// Magic bytes for metadata section
    pub const MAGIC: [u8; 4] = *b"META";
    
    /// Current metadata format version
    pub const VERSION: u8 = 1;
    
    /// Fixed size of metadata header
    pub const HEADER_SIZE: usize = 40;
    
    /// Fixed size of label array header
    pub const LABEL_ARRAY_HEADER_SIZE: usize = 8;
}

/// Structure flags for matrix properties (from existing format.rs)
pub const SYMMETRIC: u8 = 1;
pub const UPPER_TRIANGULAR: u8 = 2;
pub const LOWER_TRIANGULAR: u8 = 4;
pub const SORTED_INDICES: u8 = 8;