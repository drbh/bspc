# BSPC Architecture

BSPC builds on [binsparse-rs](https://github.com/drbh/binsparse-rs), implementing the [Binary Sparse Format Specification](https://graphblas.org/binsparse-specification/) with additional optimizations for memory-mapped access and bloom filter indexing.

## File Format

```
┌──────────┬─────────────┬────────────────────┐
│  Header  │ Bloom Filter│    Sparse Data     │
│   68B    │  Variable   │     Variable       │
└──────────┴─────────────┴────────────────────┘
```

### Header (68 bytes)
```rust
struct BspcHeader {
    magic: [u8; 4],      // "BSPC"
    version: u8,         // Format version
    format_type: u8,     // COO=0, CSR=1, CSC=2
    data_type: u8,       // f32=0, f64=1, i32=2, etc.
    nrows: u32,          // Matrix dimensions
    ncols: u32,
    nnz: u32,           // Non-zero count
    values_offset: u32,  // File offsets
    values_size: u32,
    indices_0_offset: u32,
    indices_0_size: u32,
    indices_1_offset: u32,
    indices_1_size: u32,
    // ... padding to 68 bytes
}
```

### Bloom Filter
- One filter per row chunk (configurable chunk size)
- Probabilistic membership test: "row X might be in chunk Y"
- False positives possible, false negatives impossible
- Typical size: ~10 bytes per chunk

### Sparse Data
Standard COO (Coordinate) format per Binary Sparse Format Specification:
- `values[]`: Non-zero matrix values
- `row_indices[]`: Row coordinates  
- `col_indices[]`: Column coordinates

The underlying sparse data follows the binsparse standard, wrapped with BSPC's memory-mapped interface for efficient access.

## Core Components

### MmapMatrix
- Memory-maps entire .bspc file
- Provides zero-copy slice access to arrays
- Thread-safe (Send + Sync)
- No heap allocation for data access

### ChunkedMatrix
- Splits large matrices into row chunks
- Each chunk has associated bloom filter
- Configurable memory limits
- Loads chunks on-demand

### Bloom Filter
- Fixed-size bit array per chunk
- Multiple hash functions for better accuracy
- Fast O(1) membership testing
- Enables skipping irrelevant chunks

## Query Flow

```
Query row 1000 ──┐
                 │
                 ▼
Check bloom filters:
├─ Chunk 0: ❌ Skip (no row 1000)
├─ Chunk 1: ❌ Skip (no row 1000)  
└─ Chunk 2: ✅ Load (might have row 1000)
                 │
                 ▼
Linear search within chunk 2 ──► Return results
```

## Design Decisions

**Memory mapping**: Direct file access without loading into RAM  
**Bloom filters**: Probabilistic skip optimization  
**COO format**: Simple, general sparse matrix representation  
**Fixed header**: Fast parsing without variable-length fields  
**Little-endian**: Standard byte ordering for cross-platform compatibility