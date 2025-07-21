//! Simple example to read a sparse matrix with bloom filter optimization

use bspc::{BspcFile, ChunkConfig};
use bspc_core::MatrixElement;
use std::time::Instant;

fn main() -> binsparse_rs::Result<()> {
    // let filename = "example_matrix.bspc";
    let filename = "test_v2_large.bspc";
    let start_time = Instant::now();

    // Check if file exists
    if !std::path::Path::new(filename).exists() {
        println!("File '{filename}' not found!");
        println!("   Run 'cargo run --example write_matrix' first");
        return Ok(());
    }

    println!("Reading sparse matrix from '{filename}'...");

    // Load matrix with bloom filter optimization
    let start = Instant::now();
    let config = ChunkConfig::default().with_bloom_hash_count(3);
    let matrix = BspcFile::read_matrix_with_bloom_filter(filename, config.clone())?;
    let load_time = start.elapsed();
    println!("Matrix loaded in {:.3}ms", load_time.as_secs_f64() * 1000.0);

    // Get matrix information
    let dimensions = matrix.dimensions();
    println!("\nMatrix Information:");
    println!("   Dimensions: {} x {}", dimensions.0, dimensions.1);
    println!("   Bloom filter hash count: {}", config.bloom_hash_count);

    // Test element access
    println!("\nTesting element access:");
    let test_positions = [
        (10, 0),     // Should have data
        (50, 0),     // Should have data
        (100, 0),    // Should have data
        (999, 500),  // Should be sparse
        (90000, 0),  // Should have data
        (1234, 567), // Should be sparse
        (700, 1100),
    ];

    for (row, col) in test_positions {
        if row < dimensions.0 && col < dimensions.1 {
            let start = Instant::now();
            match matrix.get_element(row, col) {
                Some(value) => {
                    let time = start.elapsed();
                    println!(
                        "   matrix[{}, {}] = {:?} (found in {:.3}ms)",
                        row,
                        col,
                        value,
                        time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let time = start.elapsed();
                    println!(
                        "   matrix[{}, {}] = 0 (sparse, {:.3}ms)",
                        row,
                        col,
                        time.as_secs_f64() * 1000.0
                    );
                }
            }
        }
    }

    // Show first 10 values of row 10 (known sparse row)
    println!("\nFirst 10 values of row 10:");
    for col in 0..10 {
        if col < dimensions.1 {
            let start = Instant::now();
            match matrix.get_element(10, col) {
                Some(value) => {
                    let time = start.elapsed();
                    println!(
                        "   [{}] = {} ({:.3}ms)",
                        col,
                        value.to_f64(),
                        time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let time = start.elapsed();
                    println!("   [{}] = 0 ({:.3}ms)", col, time.as_secs_f64() * 1000.0);
                }
            }
        }
    }

    // Show values from row 100, every 3rd column from 3 to 30 (more complex pattern)
    println!("\nRow 100, every 3rd column from 3 to 30:");
    for col in (3..=30).step_by(3) {
        if col < dimensions.1 {
            let start = Instant::now();
            match matrix.get_element(100, col) {
                Some(value) => {
                    let time = start.elapsed();
                    println!(
                        "   [{}] = {} ({:.3}ms)",
                        col,
                        value.to_f64(),
                        time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let time = start.elapsed();
                    println!("   [{}] = 0 ({:.3}ms)", col, time.as_secs_f64() * 1000.0);
                }
            }
        }
    }

    // // iterate over rows until you find a non-zero value
    // println!("\nIterating over rows until a non-zero value is found:");
    // let mut found_non_zero = false;
    // for row in 0..dimensions.0 {
    //     if found_non_zero {
    //         break;
    //     }
    //     for col in 0..dimensions.1 {
    //         let start = Instant::now();
    //         match matrix.get_element(row, col) {
    //             Some(value) => {
    //                 let time = start.elapsed();
    //                 println!(
    //                     "   matrix[{}, {}] = {} (found in {:.3}ms)",
    //                     row,
    //                     col,
    //                     value.to_f64(),
    //                     time.as_secs_f64() * 1000.0
    //                 );
    //                 if value.to_f64() > 0.1 {
    //                     println!("   Found non-zero value at matrix[{}, {}]", row, col);
    //                     found_non_zero = true;
    //                 }
    //                 break; // Stop after first non-zero value
    //             }
    //             None => {
    //                 let time = start.elapsed();
    //                 println!(
    //                     "   matrix[{}, {}] = 0 (sparse, {:.3}ms)",
    //                     row,
    //                     col,
    //                     time.as_secs_f64() * 1000.0
    //                 );
    //             }
    //         }
    //     }
    // }

    // Show total time
    let total_time = start_time.elapsed();
    println!("\nTotal time: {:.3}ms", total_time.as_secs_f64() * 1000.0);

    Ok(())
}
