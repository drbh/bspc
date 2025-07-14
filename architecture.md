# BSPC Architecture

BSPC builds on [binsparse-rs](https://github.com/drbh/binsparse-rs), implementing the [Binary Sparse Format Specification](https://graphblas.org/binsparse-specification/) with additional optimizations for memory-mapped access and bloom filter indexing.

## File Format

```
┌──────────────────────────────────────────────────────────────┐
│  BSPC Header  │  Values  │  Row Indices  │  Col Indices      │
│     68B       │ Variable │   Variable    │   Variable        │
└──────────────────────────────────────────────────────────────┘
```

### Header (68 bytes)
```rust
struct BspcHeader {
    magic: [u8; 4],         // "BSPC"
    version: u8,            // Format version
    format_type: u8,        // COO=0, CSR=1, CSC=2
    data_type: u8,          // f32=0, f64=1, i32=2, etc.
    structure_flags: u8,    // Symmetric, triangular flags
    nrows: u32,             // Matrix dimensions
    ncols: u32,
    nnz: u32,              // Non-zero count
    values_offset: u32,     // File offsets and sizes
    values_size: u32,
    indices_0_offset: u32,  // Row indices (COO)
    indices_0_size: u32,
    indices_1_offset: u32,  // Column indices (COO)
    indices_1_size: u32,
    pointers_offset: u32,   // For CSR/CSC (unused in COO)
    pointers_size: u32,
    reserved: [u8; 16],     // Future extensions
}
```

### Values Array
- Non-zero matrix values in native data type (f32, f64, i32, etc.)
- Stored in little-endian byte order
- Aligned to data type boundaries (4-byte for f32/i32, 8-byte for f64/i64)
- Size: `nnz * sizeof(data_type)` bytes

### Row Indices Array  
- Row coordinates for each non-zero value (u32)
- Stored in little-endian byte order
- Aligned to 4-byte boundaries
- Size: `nnz * 4` bytes

### Column Indices Array
- Column coordinates for each non-zero value (u32) 
- Stored in little-endian byte order
- Aligned to 4-byte boundaries
- Size: `nnz * 4` bytes

**Note**: Bloom filters are created at runtime during matrix loading, not stored in the file format.

## Core Components

### MmapMatrix
- Memory-maps entire .bspc file
- Provides zero-copy slice access to arrays
- Thread-safe (Send + Sync)
- No heap allocation for data access

### ChunkedMatrix
- Wrapper for matrices with bloom filter optimization
- Creates runtime bloom filters for efficient row queries
- Configurable memory limits and hash function count
- Provides unified API across different matrix types

### Runtime Bloom Filter
- Created during matrix loading (not stored in file)
- Fixed-size bit array (32 bytes default) 
- Configurable hash function count (1-8, default 3)
- Tracks which rows contain data for fast existence checks
- Enables skipping empty rows during queries

## Query Flow

```
Query row 1000 ──┐
                 │
                 ▼
Check runtime bloom filter:
├─ Contains row 1000? ❌ ──► Return None (definitely empty)
└─ Contains row 1000? ✅ ──┐ (might have data)
                           │
                           ▼
Linear search through COO arrays ──► Return results
```

## Design Decisions

**Memory mapping**: Direct file access without loading into RAM  
**Runtime bloom filters**: Probabilistic skip optimization without file size overhead  
**COO format**: Simple, general sparse matrix representation compatible with Binary Sparse Format  
**Fixed header**: Fast parsing without variable-length fields  
**Little-endian**: Standard byte ordering for cross-platform compatibility  
**Alignment**: Proper data alignment for direct memory access without copying  
**Zero-copy access**: Iterator-based API for efficient bulk operations