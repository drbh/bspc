//! Array bounds and alignment validation for BSPC specification
//!
//! This module provides pure mathematical validation functions for
//! array operations with no I/O dependencies.

use crate::BspcError;

/// Validate array bounds for a given element type
///
/// Performs mathematical validation of array size calculations with
/// overflow protection. This is a pure function with no I/O.
pub const fn validate_array_bounds<T>(byte_len: usize) -> Result<usize, BspcError> {
    let element_size = core::mem::size_of::<T>();

    // Check alignment
    if byte_len % element_size != 0 {
        return Err(BspcError::ArrayAlignment);
    }

    // Calculate element count with overflow protection
    let count = byte_len / element_size;

    // Conservative overflow protection - reject arrays larger than usize::MAX / 8
    // This prevents potential overflow in downstream calculations
    if count > usize::MAX / 8 {
        return Err(BspcError::ArraySizeOverflow);
    }

    Ok(count)
}

/// Validate alignment for a pointer to typed data
///
/// Checks that a raw pointer has the correct alignment for type T.
/// This is a pure mathematical check with no memory access.
/// Note: This function cannot be const due to pointer casting limitations.
pub fn validate_alignment<T>(ptr: *const u8) -> Result<(), BspcError> {
    let alignment = core::mem::align_of::<T>();
    let addr = ptr as usize;

    if addr % alignment != 0 {
        return Err(BspcError::ArrayAlignment);
    }

    Ok(())
}

/// Validate that a byte length can represent a valid array of u32 elements
///
/// This is a specialized version of validate_array_bounds for u32,
/// which is commonly used for indices in sparse matrices.
pub const fn validate_u32_array_size(byte_len: usize) -> Result<usize, BspcError> {
    validate_array_bounds::<u32>(byte_len)
}

/// Validate that a byte slice can be safely interpreted as a typed array
///
/// Combines length and alignment validation for a complete safety check.
pub fn validate_typed_slice<T>(data: &[u8]) -> Result<usize, BspcError> {
    // Validate alignment
    validate_alignment::<T>(data.as_ptr())?;

    // Validate bounds
    validate_array_bounds::<T>(data.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_array_bounds() {
        // Valid alignments
        assert_eq!(validate_array_bounds::<u32>(16), Ok(4));
        assert_eq!(validate_array_bounds::<u64>(24), Ok(3));

        // Invalid alignments
        assert_eq!(
            validate_array_bounds::<u32>(15),
            Err(BspcError::ArrayAlignment)
        );
        assert_eq!(
            validate_array_bounds::<u64>(23),
            Err(BspcError::ArrayAlignment)
        );

        // Empty arrays are valid
        assert_eq!(validate_array_bounds::<u32>(0), Ok(0));
    }

    #[test]
    fn test_validate_alignment() {
        let aligned_data: [u64; 4] = [0; 4];
        let ptr = aligned_data.as_ptr() as *const u8;

        // u64 requires 8-byte alignment, should pass
        assert_eq!(validate_alignment::<u64>(ptr), Ok(()));

        // u32 requires 4-byte alignment, should also pass (8 is multiple of 4)
        assert_eq!(validate_alignment::<u32>(ptr), Ok(()));

        // Test unaligned access
        let unaligned_ptr = unsafe { ptr.offset(1) };
        assert_eq!(
            validate_alignment::<u64>(unaligned_ptr),
            Err(BspcError::ArrayAlignment)
        );
    }
}
