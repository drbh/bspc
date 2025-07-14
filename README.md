# BSPC - Binary Sparse Container

> [!WARNING]
> This is a work in progress. The API may change before the first stable release.

<!-- [![Crates.io](https://img.shields.io/crates/v/bspc.svg)](https://crates.io/crates/bspc)
[![Documentation](https://docs.rs/bspc/badge.svg)](https://docs.rs/bspc/) -->

**Query massive sparse matrices instantly. Zero-copy. Memory-mapped.**

```
┌─────────────────────────────┐
│  Header │ Bloom │  Data     │  .bspc format
└─────────────────────────────┘
```

### Why BSPC?

Sparse matrices are everywhere in ML, but existing formats are slow for queries:

| Format   | Zero-copy | Bloom filters | Query speed |
| -------- | --------- | ------------- | ----------- |
| pickle   | ✗         | ✗             | Slow        |
| HDF5     | ✗         | ✗             | Slow        |
| NPZ      | ✗         | ✗             | Slow        |
| **BSPC** | **✓**     | **✓**         | **Fast**    |

**The secret:** Bloom filters skip 90%+ of disk reads.

### Usage

```rust
use bspc::{BspcFile, MmapMatrix};

// Save any sparse matrix
BspcFile::write_matrix(&matrix, "huge.bspc")?;

// Query without loading into memory
let mmap: MmapMatrix<f64> = BspcFile::read_matrix("huge.bspc")?;
let value = mmap.get_element(row, col)?;  // Microseconds
```

### Install

```toml
[dependencies]
bspc = "0.1"
```

**Result:** Query huge matrices on a laptop. Memory usage stays constant.


### Quickstart Guide

For a quickstart guide, see [quickstart.md](quickstart.md).

### Built on Standards

BSPC implements the [Binary Sparse Format Specification](https://graphblas.org/binsparse-specification/) through [binsparse-rs](https://github.com/drbh/binsparse-rs), adding memory-mapped access and bloom filter optimizations for instant querying.