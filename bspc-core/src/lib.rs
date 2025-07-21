#![no_std]

//! BSPC Core - Binary Sparse Matrix Format Specification
//!
//! This crate provides the complete specification for the BSPC (Binary Sparse Matrix) format,
//! including data layout definitions, abstract interfaces, and validation utilities.
//! 
//! ## Architecture
//! 
//! This crate serves as a pure specification layer with no I/O dependencies:
//! - **Format definitions**: Binary layout and wire format specifications
//! - **Abstract interfaces**: Trait definitions for implementations  
//! - **Validation utilities**: Pure mathematical validation functions
//! - **Error taxonomy**: Comprehensive error classification system
//!
//! ## Design Principles
//!
//! - **No I/O operations**: Only pure data structures and mathematical functions
//! - **No concrete implementations**: Only abstract interfaces and format definitions
//! - **No platform dependencies**: Works in no-std environments
//! - **Zero policy**: No algorithmic or business logic decisions

// Public modules
pub mod bloom_filter;
pub mod error;
pub mod format;
pub mod traits;
pub mod validation;

// Re-export core types for convenience
pub use bloom_filter::*;
pub use error::*;
pub use format::*;
pub use traits::*;
pub use validation::*;
