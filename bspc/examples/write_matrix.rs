//! Fast streaming write example for large sparse matrices with bloom filter optimization

use binsparse_rs::prelude::*;
use bspc::{BspcFile, ChunkConfig};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Writing large sparse matrix using streaming approach...");

    // Matrix dimensions - reasonable size for testing
    let nrows = 100_000_000;
    let ncols = 2_000_000;

    // Add elements to specific rows distributed throughout the matrix
    let sparse_rows = [10, 50, 100, 500, 900, 5000, 50000, 90000];
    let elements_per_row = 1_000_000;

    println!("Matrix dimensions: {nrows} x {ncols}");
    println!(
        "Sparse rows: {} rows with {} elements each",
        sparse_rows.len(),
        elements_per_row
    );
    println!("Total non-zeros: {}", sparse_rows.len() * elements_per_row);

    // Build sparse elements efficiently for demo purposes
    let start = Instant::now();
    let sparse_elements = build_demo_sparse_elements(nrows, ncols, &sparse_rows, elements_per_row);
    let build_time = start.elapsed();
    println!("Built sparse elements in {build_time:?}");

    // Write using streaming approach WITH bloom filters built during write (SUPER FAST!)
    let start = Instant::now();
    let config = ChunkConfig::default().with_bloom_hash_count(3);
    BspcFile::write_sparse_matrix_streaming(
        nrows,
        ncols,
        &sparse_elements,
        config.clone(),
        "example_matrix.bspc",
    )?;
    let write_time = start.elapsed();
    println!("Matrix + bloom filters written in {write_time:?}");
    println!("\nRun 'cargo run --example read_matrix' to read it back!");
    Ok(())
}

/// Build sparse matrix elements efficiently for demo purposes
fn build_demo_sparse_elements(
    nrows: usize,
    ncols: usize,
    sparse_rows: &[usize],
    elements_per_row: usize,
) -> Vec<(usize, usize, f64)> {
    // ) -> Vec<(usize, usize, i32)> {
    let mut elements = Vec::with_capacity(sparse_rows.len() * elements_per_row);

    for &row in sparse_rows {
        if row < nrows {
            for col in 0..elements_per_row.min(ncols) {
                let value = row as f64 + col as f64 * 0.1;
                // let value = row as f64 + col as f64;
                // let value = (row + col) as i32; // Use i16 for smaller values
                elements.push((row, col, value));
            }
        }
    }

    elements
}
