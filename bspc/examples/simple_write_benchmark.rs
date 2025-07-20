use bspc::chunked_backend::ChunkConfig;
use bspc::mmap_backend::BspcFile;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), binsparse_rs::Error> {
    println!("Simple Write Benchmark - Unified Interface");

    // Test parameters
    let nrows = 50_000;
    let ncols = 10_000;
    let nnz = 5_000_000;

    println!("Matrix: {nrows}x{ncols} with {nnz} elements");

    // Generate test data
    let data_start = Instant::now();
    let sparse_elements: Vec<(usize, usize, f32)> = (0..nnz)
        .map(|i| {
            let row = (i * 31) % nrows;
            let col = (i * 37) % ncols;
            let value = (i as f32 * 0.001) % 1000.0;
            (row, col, value)
        })
        .collect();
    println!(
        "Data generation: {:.3}s",
        data_start.elapsed().as_secs_f64()
    );

    let config = ChunkConfig::default();
    let filename = "benchmark_output.bspc";

    // Test unified interface (automatically uses fastest method)
    println!("\nWriting with unified interface...");
    let start = Instant::now();

    #[cfg(feature = "async")]
    BspcFile::write_sparse_matrix(nrows, ncols, &sparse_elements, config.clone(), filename).await?;

    #[cfg(not(feature = "async"))]
    BspcFile::write_sparse_matrix(nrows, ncols, &sparse_elements, config.clone(), filename)?;

    let duration = start.elapsed();
    let file_size = std::fs::metadata(filename)
        .map_err(|_| binsparse_rs::Error::IoError("Failed to get file size"))?
        .len();

    println!("Write completed in {:.3}s", duration.as_secs_f64());
    println!("File size: {:.1} MB", file_size as f64 / (1024.0 * 1024.0));
    println!(
        "Throughput: {:.1} MB/s",
        (file_size as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64()
    );
    println!("Elements/s: {:.0}", nnz as f64 / duration.as_secs_f64());

    // Verify the file
    println!("\nVerifying written file...");
    let verify_start = Instant::now();
    let chunked_matrix = BspcFile::read_matrix_with_bloom_filter(filename, config)?;
    let (read_nrows, read_ncols) = chunked_matrix.dimensions();
    println!("Verification: {:.3}s", verify_start.elapsed().as_secs_f64());
    println!("Dimensions: {read_nrows}x{read_ncols} (correct)");

    // Clean up
    std::fs::remove_file(filename)
        .map_err(|_| binsparse_rs::Error::IoError("Failed to remove file"))?;
    println!("Cleaned up test file");

    println!("\nBenchmark completed successfully!");

    Ok(())
}
