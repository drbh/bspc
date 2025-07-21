//! Format validation utilities for BSPC specification
//!
//! This module contains pure validation functions with no I/O dependencies.
//! All functions are mathematical operations on data layout and format constraints.

pub mod bounds;
pub mod format;
pub mod parsing;

pub use bounds::{validate_array_bounds, validate_alignment};
pub use format::align_to_boundary;
pub use parsing::parse_range;