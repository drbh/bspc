# BSPC - Binary Sparse Container

> [!WARNING]
> This is a work in progress. The API may change before the first stable release.

<!-- [![Crates.io](https://img.shields.io/crates/v/bspc.svg)](https://crates.io/crates/bspc)
[![Documentation](https://docs.rs/bspc/badge.svg)](https://docs.rs/bspc/) -->

**A memory-mapped sparse matrix format with runtime bloom filter optimization.**

BSPC implements the Binary Sparse Format standard with memory-mapped access and optional bloom filters for faster row queries on large matrices.

### Usage

```rust
use bspc::{BspcFile, MmapMatrix};

// Save a sparse matrix
BspcFile::write_matrix(&matrix, "data.bspc")?;

// Memory-map and query
let mmap: MmapMatrix<f64> = BspcFile::read_matrix("data.bspc")?;
let value = mmap.get_element(row, col)?;
```

### Install

```toml
[dependencies]
bspc = "0.1"
```

Query large matrices without loading them into memory.


### Quickstart Guide

For a quickstart guide, see [quickstart.md](quickstart.md).

### Built on Standards

BSPC implements the [Binary Sparse Format Specification](https://graphblas.org/binsparse-specification/) through [binsparse-rs](https://github.com/drbh/binsparse-rs), adding memory-mapped access and bloom filter optimizations for instant querying.