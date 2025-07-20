//! Example showing streaming access pattern for remote file stores like S3
//!
//! This demonstrates how to efficiently access sparse matrix data by:
//! 1. Loading metadata first (fast header read)
//! 2. Streaming only the needed rows on-demand
//! 3. Using bloom filters to optimize row existence checks

use bspc::{BspcFile, ChunkConfig};
use std::time::Instant;

fn main() -> binsparse_rs::Result<()> {
    let filename = "example_matrix.bspc";

    // Check if file exists
    if !std::path::Path::new(filename).exists() {
        println!("File '{filename}' not found!");
        println!("   Run 'cargo run --example write_matrix' first");
        return Ok(());
    }

    println!("=== Streaming Access Pattern Demo ===");
    println!("Simulating remote file store access (S3, etc.)\n");

    // Step 1: Fast metadata-only access (like S3 HEAD request)
    println!("1. Loading metadata only...");
    let start = Instant::now();
    let config = ChunkConfig::default().with_bloom_hash_count(3);
    let matrix = BspcFile::read_matrix_with_bloom_filter(filename, config)?;
    let metadata_time = start.elapsed();

    let dimensions = matrix.dimensions();
    println!(
        "   Metadata loaded in {:.3}ms",
        metadata_time.as_secs_f64() * 1000.0
    );
    println!(
        "   Matrix: {} x {} with bloom filter optimization",
        dimensions.0, dimensions.1
    );

    // Step 2: Stream row ranges (chunks) - like S3 range requests
    let row_ranges = [(0, 100), (100, 200), (200, 300), (500, 600), (1000, 1100)];
    println!("\n2. Streaming {} row ranges...", row_ranges.len());

    let batch_start = Instant::now();
    let mut total_values = 0;

    for (start_row, end_row) in row_ranges {
        println!("   Range {start_row}-{end_row}: ");
        let range_start = Instant::now();
        let mut range_values = 0;

        // Stream all rows in this range using optimized bulk row processing
        if let Ok(range_iter) = matrix.row_range_view(start_row, end_row.min(dimensions.0)) {
            let mut row_counts = std::collections::HashMap::new();

            // Process all values in the range in a single pass
            for (row, col, _value) in range_iter {
                // Only count first 3 columns per row
                if col < 3 {
                    *row_counts.entry(row).or_insert(0) += 1;
                }
            }

            // Display results
            for (row_idx, count) in row_counts.iter() {
                if *count > 0 {
                    println!("     Row {row_idx} has {} values", count.min(&3));
                    range_values += count.min(&3);
                    total_values += count.min(&3);
                }
            }
        }

        let range_time = range_start.elapsed();
        println!(
            "     → {} values in {:.3}ms",
            range_values,
            range_time.as_secs_f64() * 1000.0
        );
    }

    let batch_time = batch_start.elapsed();
    println!(
        "   → Total: {} values across {} ranges in {:.3}ms",
        total_values,
        row_ranges.len(),
        batch_time.as_secs_f64() * 1000.0
    );
    println!(
        "   → Average: {:.3}ms per range",
        batch_time.as_secs_f64() * 1000.0 / row_ranges.len() as f64
    );

    // Step 3: Bulk sparse row detection (bloom filter optimization)
    println!("\n3. Bulk sparse row detection:");
    let sparse_test_rows = [999, 1234, 5678, 12345, 99999, 123456];
    let sparse_start = Instant::now();

    let mut sparse_count = 0;
    let mut data_count = 0;

    for &row in &sparse_test_rows {
        if row < dimensions.0 {
            if let Ok(mut row_iter) = matrix.row_view(row) {
                if let Some((_, value)) = row_iter.next() {
                    println!("   Row {row} has data: {value}");
                    data_count += 1;
                } else {
                    sparse_count += 1;
                }
            } else {
                sparse_count += 1;
            }
        } else {
            sparse_count += 1;
        }
    }

    let sparse_total = sparse_start.elapsed();
    println!("   → Found {data_count} data rows, {sparse_count} sparse rows");
    println!(
        "   → Checked {} rows in {:.3}ms total",
        sparse_test_rows.len(),
        sparse_total.as_secs_f64() * 1000.0
    );
    println!(
        "   → {:.3}ms per row check",
        sparse_total.as_secs_f64() * 1000.0 / sparse_test_rows.len() as f64
    );

    // Summary
    let total_time = start.elapsed();
    println!("\n=== Summary ===");
    println!(
        "Total streaming session: {:.3}ms",
        total_time.as_secs_f64() * 1000.0
    );
    println!(
        "Metadata overhead: {:.1}%",
        (metadata_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );

    Ok(())
}
