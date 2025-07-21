//! Parsing utilities for BSPC format strings
//!
//! This module provides pure parsing functions for BSPC format-specific
//! string representations with no I/O dependencies.

use crate::BspcError;
use core::ops::Range;

/// Parse a range string in the format "start:end" or "start-end"
///
/// This function parses common range representations used in BSPC
/// contexts (like HTTP Range headers). Returns a Range<usize>.
pub fn parse_range(range_str: &str) -> Result<Range<usize>, BspcError> {
    // Handle empty string
    if range_str.is_empty() {
        return Err(BspcError::InvalidRange);
    }

    // Try colon separator first
    if let Some(colon_pos) = range_str.find(':') {
        let start_str = &range_str[..colon_pos];
        let end_str = &range_str[colon_pos + 1..];

        let start = parse_usize(start_str)?;
        let end = parse_usize(end_str)?;

        if start > end {
            return Err(BspcError::InvalidRange);
        }

        return Ok(start..end);
    }

    // Try dash separator
    if let Some(dash_pos) = range_str.find('-') {
        let start_str = &range_str[..dash_pos];
        let end_str = &range_str[dash_pos + 1..];

        let start = parse_usize(start_str)?;
        let end = parse_usize(end_str)?;

        if start > end {
            return Err(BspcError::InvalidRange);
        }

        return Ok(start..end);
    }

    // No separator found
    Err(BspcError::InvalidRange)
}

/// Parse a usize from a string with error handling
///
/// Helper function for parsing numeric strings with proper error mapping.
fn parse_usize(s: &str) -> Result<usize, BspcError> {
    // Handle empty strings
    if s.is_empty() {
        return Err(BspcError::InvalidRange);
    }

    // Parse manually to avoid std dependency
    let mut result: usize = 0;

    for byte in s.bytes() {
        if !byte.is_ascii_digit() {
            return Err(BspcError::InvalidRange);
        }

        let digit = (byte - b'0') as usize;

        // Check for overflow
        if result > (usize::MAX - digit) / 10 {
            return Err(BspcError::ArraySizeOverflow);
        }

        result = result * 10 + digit;
    }

    Ok(result)
}

/// Parse a version string in the format "major.minor.patch"
///
/// Returns (major, minor, patch) tuple. Patch version is optional.
pub fn parse_version(version_str: &str) -> Result<(u8, u8, u8), BspcError> {
    if version_str.is_empty() {
        return Err(BspcError::InvalidRange);
    }

    let parts = version_str.split('.');
    let mut version_parts = [0u8; 3];
    let mut count = 0;

    for part in parts {
        if count >= 3 {
            return Err(BspcError::InvalidRange); // Too many parts
        }

        if part.is_empty() {
            return Err(BspcError::InvalidRange);
        }

        let mut num: u8 = 0;
        for byte in part.bytes() {
            if !byte.is_ascii_digit() {
                return Err(BspcError::InvalidRange);
            }

            let digit = byte - b'0';

            // Check for overflow in u8
            if num > (255 - digit) / 10 {
                return Err(BspcError::ArraySizeOverflow);
            }

            num = num * 10 + digit;
        }

        version_parts[count] = num;
        count += 1;
    }

    if count < 2 {
        return Err(BspcError::InvalidRange); // Need at least major.minor
    }

    Ok((version_parts[0], version_parts[1], version_parts[2]))
}

/// Validate a label string for BSPC metadata
///
/// Checks that a label contains only valid characters and is within
/// reasonable length limits.
pub fn validate_label(label: &str) -> Result<(), BspcError> {
    if label.is_empty() {
        return Err(BspcError::InvalidLabel);
    }

    // Check length limit (reasonable for labels)
    if label.len() > 1024 {
        return Err(BspcError::InvalidLabel);
    }

    // Check for null bytes (problematic in C-style strings)
    if label.bytes().any(|b| b == 0) {
        return Err(BspcError::InvalidLabel);
    }

    // Check for control characters (except tab, newline)
    for byte in label.bytes() {
        if byte < 32 && byte != b'\t' && byte != b'\n' {
            return Err(BspcError::InvalidLabel);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range() {
        // Valid colon format
        assert_eq!(parse_range("0:10"), Ok(0..10));
        assert_eq!(parse_range("5:15"), Ok(5..15));
        assert_eq!(parse_range("100:200"), Ok(100..200));

        // Valid dash format
        assert_eq!(parse_range("0-10"), Ok(0..10));
        assert_eq!(parse_range("5-15"), Ok(5..15));

        // Invalid cases
        assert_eq!(parse_range(""), Err(BspcError::InvalidRange));
        assert_eq!(parse_range("10:5"), Err(BspcError::InvalidRange)); // start > end
        assert_eq!(parse_range("abc:def"), Err(BspcError::InvalidRange));
        assert_eq!(parse_range("10"), Err(BspcError::InvalidRange)); // no separator
        assert_eq!(parse_range("10:"), Err(BspcError::InvalidRange)); // empty end
        assert_eq!(parse_range(":10"), Err(BspcError::InvalidRange)); // empty start
    }

    #[test]
    fn test_parse_usize() {
        assert_eq!(parse_usize("0"), Ok(0));
        assert_eq!(parse_usize("123"), Ok(123));
        assert_eq!(parse_usize("999999"), Ok(999999));

        // Invalid cases
        assert_eq!(parse_usize(""), Err(BspcError::InvalidRange));
        assert_eq!(parse_usize("abc"), Err(BspcError::InvalidRange));
        assert_eq!(parse_usize("12a"), Err(BspcError::InvalidRange));
        assert_eq!(parse_usize("-123"), Err(BspcError::InvalidRange));
    }

    #[test]
    fn test_parse_version() {
        assert_eq!(parse_version("1.0.0"), Ok((1, 0, 0)));
        assert_eq!(parse_version("2.5.10"), Ok((2, 5, 10)));
        assert_eq!(parse_version("1.0"), Ok((1, 0, 0))); // patch defaults to 0

        // Invalid cases
        assert_eq!(parse_version(""), Err(BspcError::InvalidRange));
        assert_eq!(parse_version("1"), Err(BspcError::InvalidRange));
        assert_eq!(parse_version("1.0.0.0"), Err(BspcError::InvalidRange));
        assert_eq!(parse_version("a.b.c"), Err(BspcError::InvalidRange));
        assert_eq!(parse_version("1..0"), Err(BspcError::InvalidRange));
    }

    #[test]
    fn test_validate_label() {
        // Valid labels
        assert_eq!(validate_label("test"), Ok(()));
        assert_eq!(validate_label("row_123"), Ok(()));
        assert_eq!(validate_label("column-name"), Ok(()));
        assert_eq!(validate_label("label with spaces"), Ok(()));
        assert_eq!(validate_label("label\twith\ttabs"), Ok(()));

        // Invalid labels
        assert_eq!(validate_label(""), Err(BspcError::InvalidLabel));
        assert_eq!(
            validate_label("label\0with\0nulls"),
            Err(BspcError::InvalidLabel)
        );
        assert_eq!(
            validate_label("label\x01with\x02control"),
            Err(BspcError::InvalidLabel)
        );

        // Very long label should fail
        let long_label = "a".repeat(2000);
        assert_eq!(validate_label(&long_label), Err(BspcError::InvalidLabel));
    }
}
