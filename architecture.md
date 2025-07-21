# BSPC Architecture

BSPC builds on [binsparse-rs](https://github.com/drbh/binsparse-rs), implementing the [Binary Sparse Format Specification](https://graphblas.org/binsparse-specification/) with additional optimizations for memory-mapped access and bloom filter indexing.

## File Format

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BSPC Header  │  Values  │  Row Indices  │  Col Indices  │  Metadata    │
│     160B      │ Variable │   Variable    │   Variable    │  Optional    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Header (160 bytes)
```rust
struct BspcHeader {
    magic: [u8; 4],             // "BSPC"
    version: u8,                // Format version
    format_type: u8,            // COO=0, CSR=1, CSC=2
    data_type: u8,              // f32=0, f64=1, i32=2, etc.
    structure_flags: u8,        // Symmetric, triangular flags
    nrows: u64,                 // Matrix dimensions
    ncols: u64,
    nnz: u64,                   // Non-zero count
    values_offset: u64,         // File offsets and sizes
    values_size: u64,
    indices_0_offset: u64,      // Row indices (COO)
    indices_0_size: u64,
    indices_1_offset: u64,      // Column indices (COO)
    indices_1_size: u64,
    pointers_offset: u64,       // For CSR/CSC (unused in COO)
    pointers_size: u64,
    metadata_offset: u64,       // Optional metadata region
    metadata_size: u64,
    bloom_filter_offset: u64,   // Optional stored bloom filter
    bloom_filter_size: u64,
    reserved: [u8; 32],         // Future extensions
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

### Metadata Section (Optional)
- Arbitrary metadata stored as bytes
- Application-specific information
- Only present if metadata_size > 0

### Bloom Filter Section (Optional)
- Pre-computed bloom filter for row existence checks
- Stored in file for faster startup
- Only present if bloom_filter_size > 0

## System Architecture

```
┌─────────────────┐
│      bspc       │  ← I/O layer
│   (file I/O)    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   bspc-core     │  ← Core library
│  (data access)  │
└─────────────────┘
```

### bspc-core
Foundation library providing:
- **Backend abstraction**: File system, HTTP, and memory-mapped data sources
- **MmapMatrix**: Zero-copy memory-mapped matrix access
- **ChunkedMatrix**: Bloom filter wrapper for optimized row queries  
- **Runtime Bloom Filter**: Probabilistic row existence checking

### bspc
File I/O layer built on bspc-core:
- **BspcFile**: Read/write .bspc files to disk
- **Serialization**: Convert matrices to/from BSPC format
- **Type safety**: Compile-time guarantees for data types

## Dependency Flow

```
User Code
    │
    ▼
bspc::BspcFile ──► bspc_core::MmapMatrix ──► bspc_core::Backend
    │                      │                      │
    │                      ▼                      ▼
    └──────────► bspc_core::ChunkedMatrix    File/HTTP/Memory
                      │
                      ▼
                bspc_core::BloomFilter
```

## Query Flow

```
Query (row, col) ──┐
                   │
                   ▼
Check bloom filter (if available):
├─ Row exists? ❌ ──► Return None (row definitely empty)
└─ Row exists? ✅ ──┐ (row might have data)
                    │
                    ▼
Binary/linear search through indices ──► Return value or None
```

## Design Decisions

**Memory mapping**: Direct file access without loading entire matrix into RAM  
**Bloom filters**: Probabilistic row existence checks (can be stored or computed at runtime)  
**COO format**: Simple, general sparse matrix representation  
**Fixed header**: 160-byte header with u64 offsets for large file support  
**Little-endian**: Standard byte ordering for cross-platform compatibility  
**u32 indices**: Balance between memory efficiency and matrix size support (up to 4.3B × 4.3B)  
**Backend abstraction**: Supports local files, HTTP endpoints, and custom data sources