//! Simple example to read a sparse matrix from a .bspc file

use bspc::BspcFile;
use std::time::Instant;

fn main() -> binsparse_rs::Result<()> {
    let filename = "example_matrix.bspc";

    // Check if file exists
    if !std::path::Path::new(filename).exists() {
        println!("File '{filename}' not found!");
        println!("   Run 'cargo run --example write_matrix' first");
        return Ok(());
    }

    println!("Reading sparse matrix from '{filename}'...");

    // Time file opening and header parsing
    println!("Opening file and parsing header...");
    let start = Instant::now();
    let mmap_matrix = BspcFile::read_matrix_dynamic(filename)?;
    let load_time = start.elapsed();
    println!("Header parsed in {:.3}ms", load_time.as_secs_f64() * 1000.0);

    // Time matrix metadata access
    println!("Accessing matrix metadata...");
    let start_meta = Instant::now();
    let dimensions = (mmap_matrix.nrows(), mmap_matrix.ncols());
    let nnz = mmap_matrix.nnz();
    let format = mmap_matrix.format();
    let data_type = mmap_matrix.data_type();
    let meta_time = start_meta.elapsed();
    println!(
        "Metadata accessed in {:.3}ms",
        meta_time.as_secs_f64() * 1000.0
    );

    println!("\nMatrix Information:");
    println!("   Dimensions: {} x {}", dimensions.0, dimensions.1);
    println!("   Non-zeros: {nnz}");
    println!("   Format: {format}");
    println!("   Data type: {data_type}");
    println!(
        "   Sparsity: {:.6}%",
        (nnz as f64 / (dimensions.0 * dimensions.1) as f64) * 100.0
    );

    // Time element access operations with smart selection
    println!("\nTesting element access performance:");

    // Test known positions that should have values (from the generation pattern)
    println!("Testing likely positions (based on generation pattern):");
    let likely_positions = [(0, 0), (7, 0), (14, 0), (21, 0)]; // Pattern from TestMatrixBuilder

    for (i, (row, col)) in likely_positions.iter().enumerate() {
        if *row < dimensions.0 && *col < dimensions.1 {
            print!("   Access {}: matrix[{}, {}] → ", i + 1, row, col);
            let start_lookup = Instant::now();
            match mmap_matrix.get_element(*row, *col)? {
                Some(value) => {
                    let lookup_time = start_lookup.elapsed();
                    // Format the value based on its actual type to show proper decimals
                    let formatted_value = match &value {
                        binsparse_rs::array::ArrayValue::Float64(v) => format!("{v:.1}"),
                        binsparse_rs::array::ArrayValue::Float32(v) => format!("{v:.1}"),
                        _ => format!("{value}"),
                    };
                    println!(
                        "{} (HIT in {:.3}ms)",
                        formatted_value,
                        lookup_time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let lookup_time = start_lookup.elapsed();
                    println!("0.0 (MISS in {:.3}ms)", lookup_time.as_secs_f64() * 1000.0);
                }
            }
        }
    }

    // Test a few random positions (these will likely miss in sparse matrix)
    println!("\nTesting random positions (likely sparse):");
    let random_positions = [(500, 500), (90000, 500), (50000, 250)];

    for (i, (row, col)) in random_positions.iter().enumerate() {
        if *row < dimensions.0 && *col < dimensions.1 {
            print!("   Random {}: matrix[{}, {}] → ", i + 1, row, col);
            let start_lookup = Instant::now();
            match mmap_matrix.get_element(*row, *col)? {
                Some(value) => {
                    let lookup_time = start_lookup.elapsed();
                    // Format the value based on its actual type to show proper decimals
                    let formatted_value = match &value {
                        binsparse_rs::array::ArrayValue::Float64(v) => format!("{v:.1}"),
                        binsparse_rs::array::ArrayValue::Float32(v) => format!("{v:.1}"),
                        _ => format!("{value}"),
                    };
                    println!(
                        "{} (HIT in {:.3}ms)",
                        formatted_value,
                        lookup_time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let lookup_time = start_lookup.elapsed();
                    println!(
                        "0.0 (SPARSE in {:.3}ms)",
                        lookup_time.as_secs_f64() * 1000.0
                    );
                }
            }
        }
    }

    // Use lazy row view to access first 10 values of the 10th row
    println!("\nLazy access to first 10 values of row 10:");
    let row_index = 90000;
    if row_index < dimensions.0 {
        let start_row = Instant::now();
        let row_view = mmap_matrix.row_view(row_index)?;
        let row_time = start_row.elapsed();

        println!(
            "   Row {}: {} non-zero elements (view created in {:.3}ms)",
            row_index,
            row_view.len(),
            row_time.as_secs_f64() * 1000.0
        );

        // Access first 10 column positions lazily
        println!("   First 10 column values:");
        for col in 0..10.min(dimensions.1) {
            let start_access = Instant::now();
            match row_view.get_element(0, col)? {
                Some(value) => {
                    let access_time = start_access.elapsed();
                    // Display the value based on its actual type
                    println!(
                        "     [{}] = {} (in {:.3}ms)",
                        col,
                        value,
                        access_time.as_secs_f64() * 1000.0
                    );
                }
                None => {
                    let access_time = start_access.elapsed();
                    println!(
                        "     [{}] = 0 (sparse, in {:.3}ms)",
                        col,
                        access_time.as_secs_f64() * 1000.0
                    );
                }
            }
        }
    } else {
        println!(
            "   Row index {} out of bounds (max row index: {})",
            row_index,
            dimensions.0 - 1
        );
    }

    // Show total time
    let total_time = load_time + meta_time;
    println!(
        "\nTotal operation time: {:.3}ms",
        total_time.as_secs_f64() * 1000.0
    );

    Ok(())
}
