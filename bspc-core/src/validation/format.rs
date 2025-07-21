//! Format-specific validation utilities for BSPC specification
//!
//! This module provides pure validation functions for BSPC format
//! constraints and layout requirements.

use crate::BspcError;

/// Align an offset to a specific boundary
/// 
/// This is a pure mathematical function that calculates the next
/// aligned offset for a given alignment boundary.
pub const fn align_to_boundary(offset: usize, boundary: usize) -> usize {
    (offset + boundary - 1) & !(boundary - 1)
}

/// Align an offset to 8-byte boundary (common BSPC alignment)
/// 
/// Convenience function for the most common alignment requirement
/// in BSPC format.
pub const fn align_to_8(offset: usize) -> usize {
    align_to_boundary(offset, 8)
}

/// Validate that a boundary is a power of 2
/// 
/// Alignment boundaries must be powers of 2 for efficient calculation.
pub const fn validate_alignment_boundary(boundary: usize) -> Result<(), BspcError> {
    if boundary == 0 || (boundary & (boundary - 1)) != 0 {
        return Err(BspcError::InvalidRange);
    }
    Ok(())
}

/// Calculate padding needed to reach alignment boundary
/// 
/// Returns the number of bytes that must be added to reach the
/// next alignment boundary.
pub const fn calculate_padding(offset: usize, boundary: usize) -> usize {
    let aligned = align_to_boundary(offset, boundary);
    aligned - offset
}

/// Validate that an offset is properly aligned
/// 
/// Checks that an offset meets the specified alignment requirement.
pub const fn validate_offset_alignment(offset: usize, boundary: usize) -> Result<(), BspcError> {
    if offset % boundary != 0 {
        return Err(BspcError::ArrayAlignment);
    }
    Ok(())
}

/// Validate chunk boundary constraints
/// 
/// Ensures that chunk boundaries are valid and don't overflow.
pub const fn validate_chunk_boundaries(start: usize, end: usize, total_size: usize) -> Result<(), BspcError> {
    // Start must be less than or equal to end
    if start > end {
        return Err(BspcError::InvalidRange);
    }
    
    // End must not exceed total size
    if end > total_size {
        return Err(BspcError::IndexOutOfBounds);
    }
    
    Ok(())
}

/// Validate magic bytes match expected pattern
/// 
/// Compares magic bytes with expected pattern in a const-friendly way.
pub const fn validate_magic_bytes(actual: &[u8; 4], expected: &[u8; 4]) -> Result<(), BspcError> {
    if actual[0] != expected[0] 
        || actual[1] != expected[1]
        || actual[2] != expected[2] 
        || actual[3] != expected[3] {
        return Err(BspcError::InvalidHeader);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_boundary() {
        assert_eq!(align_to_boundary(0, 8), 0);
        assert_eq!(align_to_boundary(1, 8), 8);
        assert_eq!(align_to_boundary(7, 8), 8);
        assert_eq!(align_to_boundary(8, 8), 8);
        assert_eq!(align_to_boundary(9, 8), 16);
        
        assert_eq!(align_to_boundary(0, 4), 0);
        assert_eq!(align_to_boundary(3, 4), 4);
        assert_eq!(align_to_boundary(5, 4), 8);
    }

    #[test]
    fn test_align_to_8() {
        assert_eq!(align_to_8(0), 0);
        assert_eq!(align_to_8(1), 8);
        assert_eq!(align_to_8(7), 8);
        assert_eq!(align_to_8(8), 8);
        assert_eq!(align_to_8(15), 16);
    }

    #[test]
    fn test_validate_alignment_boundary() {
        // Valid power-of-2 boundaries
        assert_eq!(validate_alignment_boundary(1), Ok(()));
        assert_eq!(validate_alignment_boundary(2), Ok(()));
        assert_eq!(validate_alignment_boundary(4), Ok(()));
        assert_eq!(validate_alignment_boundary(8), Ok(()));
        assert_eq!(validate_alignment_boundary(16), Ok(()));
        
        // Invalid boundaries
        assert_eq!(validate_alignment_boundary(0), Err(BspcError::InvalidRange));
        assert_eq!(validate_alignment_boundary(3), Err(BspcError::InvalidRange));
        assert_eq!(validate_alignment_boundary(5), Err(BspcError::InvalidRange));
        assert_eq!(validate_alignment_boundary(6), Err(BspcError::InvalidRange));
        assert_eq!(validate_alignment_boundary(7), Err(BspcError::InvalidRange));
    }

    #[test]
    fn test_calculate_padding() {
        assert_eq!(calculate_padding(0, 8), 0);
        assert_eq!(calculate_padding(1, 8), 7);
        assert_eq!(calculate_padding(7, 8), 1);
        assert_eq!(calculate_padding(8, 8), 0);
        assert_eq!(calculate_padding(9, 8), 7);
    }

    #[test]
    fn test_validate_chunk_boundaries() {
        // Valid boundaries
        assert_eq!(validate_chunk_boundaries(0, 10, 20), Ok(()));
        assert_eq!(validate_chunk_boundaries(5, 15, 20), Ok(()));
        assert_eq!(validate_chunk_boundaries(0, 20, 20), Ok(()));
        assert_eq!(validate_chunk_boundaries(10, 10, 20), Ok(()));
        
        // Invalid boundaries
        assert_eq!(validate_chunk_boundaries(15, 10, 20), Err(BspcError::InvalidRange));
        assert_eq!(validate_chunk_boundaries(0, 25, 20), Err(BspcError::IndexOutOfBounds));
        assert_eq!(validate_chunk_boundaries(25, 30, 20), Err(BspcError::IndexOutOfBounds));
    }

    #[test]
    fn test_validate_magic_bytes() {
        let magic1 = b"TEST";
        let magic2 = b"TEST";
        let magic3 = b"FAIL";
        
        assert_eq!(validate_magic_bytes(magic1, magic2), Ok(()));
        assert_eq!(validate_magic_bytes(magic1, magic3), Err(BspcError::InvalidHeader));
    }
}