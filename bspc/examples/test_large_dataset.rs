use bspc::chunked_backend::ChunkConfig;
use bspc::mmap_backend::BspcFile;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), binsparse_rs::Error> {
    println!("Testing Large Dataset Support with V2 Format");

    // Test parameters - simulate the problematic dataset size
    let nrows = 221_273;
    let ncols = 18_080;
    let target_nnz = 1_932_554_688; // Size that crashed before
                                    // let target_nnz = 1_000_000_000; // Size that crashed before

    println!("Simulating large dataset:");
    println!("  Rows: {nrows}");
    println!("  Cols: {ncols}");
    println!("  Target elements: {target_nnz}");

    // Create a smaller representative dataset for testing
    // let test_nnz = 1_932_554_688; // 10M elements for testing
    let test_nnz = target_nnz; // 10M elements for testing
    println!("  Actual test elements: {test_nnz}");

    // Generate test data
    println!("Generating sparse matrix data...");
    let sparse_elements: Vec<(usize, usize, f32)> = (0..test_nnz)
        .map(|i| {
            let row = (i * 7) % nrows;
            let col = (i * 11) % ncols;
            let value = (i as f32) / 1000.0;
            (row, col, value)
        })
        .collect();

    // Configuration for V2 format
    let config = ChunkConfig {
        memory_limit_mb: 2048,
        bloom_hash_count: 3,
        chunk_size: 100_000,
    };

    println!("Writing with V2 format (u64 support)...");
    let start = Instant::now();

    // Use unified interface (automatically selects fastest method)
    #[cfg(feature = "async")]
    BspcFile::write_sparse_matrix(nrows, ncols, &sparse_elements, config, "test_v2_large.bspc")
        .await?;

    #[cfg(not(feature = "async"))]
    BspcFile::write_sparse_matrix(nrows, ncols, &sparse_elements, config, "test_v2_large.bspc")?;

    let duration = start.elapsed();

    println!(
        "Successfully wrote {} elements in {:.3}s",
        test_nnz,
        duration.as_secs_f64()
    );

    println!(
        "Throughput: {:.1} elements/second",
        test_nnz as f64 / duration.as_secs_f64()
    );
    println!(
        "Data rate: {:.1} MB/s",
        (test_nnz * 12) as f64 / (1024.0 * 1024.0) / duration.as_secs_f64()
    );

    // Calculate theoretical capacity
    let bytes_per_element = 4 + 4 + 4; // f32 value + u32 row + u32 col
    let u64_max_elements = u64::MAX / bytes_per_element as u64;

    println!("\nV2 Format Capabilities:");
    println!("   - u32 limit (old): ~1.07B elements");
    println!(
        "   - u64 limit (new): ~{:.1}B elements",
        u64_max_elements as f64 / 1e9
    );
    println!("   - Target dataset: 857M elements (now supported)");

    // Clean up
    // std::fs::remove_file("test_v2_large.bspc").ok();

    Ok(())
}
