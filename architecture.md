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

### Bloom Filter Section
- Pre-computed bloom filter for row existence checks
- If bloom_filter_size == 0, computed at runtime during load
- Always used for consistent query performance

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
- **SparseMatrix trait**: Backend abstraction for different data sources
- **BloomFilter64**: Probabilistic data structure primitives
- **Format definitions**: Header structures and validation
- **Core traits**: Building blocks for matrix implementations

### bspc
Implementation layer built on bspc-core:
- **ChunkedMatrix<M>**: Generic wrapper with bloom filter optimization
- **ChunkBloomFilter**: Chunk-based bloom filters (wraps BloomFilter64)
- **MmapMatrix<T>**: Memory-mapped matrix implementations
- **BspcFile**: File I/O operations and format serialization

## Dependency Flow

```
User Code
    │
    ▼
bspc::BspcFile
    │
    ▼
bspc::ChunkedMatrix<M>
    │
    ├─► M: SparseMatrix (DynamicMatrix, MmapMatrix<T>)
    │       │
    │       ▼
    │   Backend abstraction ──► File/HTTP/Memory
    │
    └─► bspc::ChunkBloomFilter
            │
            ▼
        bspc_core::BloomFilter64
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
**Bloom filters**: Always-on probabilistic filters for consistent performance (computed at runtime if not stored)  
**COO format**: Simple, general sparse matrix representation  
**Fixed header**: 160-byte header with u64 offsets for large file support  
**Little-endian**: Standard byte ordering for cross-platform compatibility  
**u32 indices**: Balance between memory efficiency and matrix size support (up to 4.3B × 4.3B)  
**Backend abstraction**: Supports local files, HTTP endpoints, and custom data sources