//! Matrix element type constraints for BSPC specification
//!
//! This module defines the trait that constrains what types can be
//! stored as matrix elements in BSPC format.

use crate::format::DataType;

/// Trait for types that can be stored as matrix elements
///
/// This trait defines the requirements for types that can be stored
/// in sparse matrices. All matrix element types must be:
/// - Copy: Can be copied without allocation
/// - Clone: Can be cloned
/// - PartialEq: Can be compared for equality
/// - Sized: Have a known size at compile time
pub trait MatrixElement: Copy + Clone + PartialEq + Sized {
    /// Get the BSPC DataType representation for this element type
    fn data_type() -> DataType;

    /// Get the size in bytes of this element type
    fn size_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Convert from f64 for generic construction
    ///
    /// This is used for generic matrix construction where the exact
    /// element type may not be known at compile time.
    fn from_f64(value: f64) -> Self;

    /// Convert to f64 for generic operations
    ///
    /// This is used for generic operations where a common numeric
    /// type is needed.
    fn to_f64(self) -> f64;
}

// Implement MatrixElement for standard numeric types

impl MatrixElement for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MatrixElement for f64 {
    fn data_type() -> DataType {
        DataType::F64
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl MatrixElement for i32 {
    fn data_type() -> DataType {
        DataType::I32
    }

    fn from_f64(value: f64) -> Self {
        value as i32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MatrixElement for i64 {
    fn data_type() -> DataType {
        DataType::I64
    }

    fn from_f64(value: f64) -> Self {
        value as i64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MatrixElement for u32 {
    fn data_type() -> DataType {
        DataType::U32
    }

    fn from_f64(value: f64) -> Self {
        value as u32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MatrixElement for u64 {
    fn data_type() -> DataType {
        DataType::U64
    }

    fn from_f64(value: f64) -> Self {
        value as u64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

// Note: ArrayValue from binsparse_rs doesn't implement Copy, so it can't
// directly implement MatrixElement. Use wrapper types or specific conversions instead.
