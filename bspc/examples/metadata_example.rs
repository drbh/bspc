//! Example showing how to use metadata functionality in BSPC files
//!
//! This example demonstrates writing and reading matrices with structured metadata
//! including row and column labels with O(1) lookups.

use bspc::{chunked_backend::ChunkConfig, mmap_backend::BspcFile};

#[tokio::main]
async fn main() -> Result<(), binsparse_rs::Error> {
    // Create test data
    let sparse_elements = vec![
        (0, 0, 1.0_f64),
        (0, 2, 2.0_f64),
        (1, 1, 3.0_f64),
        (2, 0, 4.0_f64),
        (2, 2, 5.0_f64),
    ];

    // Create labels
    let row_labels = [
        b"gene_A\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
        b"gene_B\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
        b"gene_C\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
    ];

    let col_labels = [
        b"sample_1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
        b"sample_2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
        b"sample_3\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", // 32 bytes
    ];

    // Convert to &[u8] slices
    let row_label_refs: Vec<&[u8]> = row_labels.iter().map(|x| x.as_slice()).collect();
    let col_label_refs: Vec<&[u8]> = col_labels.iter().map(|x| x.as_slice()).collect();

    let filename = "test_metadata.bspc";
    let config = ChunkConfig::default();

    // Write matrix with metadata
    BspcFile::write_sparse_matrix_with_labels(
        3, // nrows
        3, // ncols
        &sparse_elements,
        &row_label_refs,
        &col_label_refs,
        32, // label_stride
        config.clone(),
        filename,
    )
    .await?;

    // Read the matrix back
    let chunked_matrix = BspcFile::read_matrix_with_bloom_filter(filename, config)?;
    let matrix = chunked_matrix.matrix();

    // Test metadata access
    println!("Matrix dimensions: {}x{}", matrix.nrows(), matrix.ncols());
    println!("Non-zero elements: {}", matrix.nnz());

    // Test raw metadata access
    if let Some(metadata_bytes) = matrix.metadata_bytes() {
        println!("Metadata size: {} bytes", metadata_bytes.len());

        // Test structured metadata access
        if let Some(metadata_view) = matrix.metadata_view()? {
            // Test row labels
            println!("Row labels:");
            if let Some(row_labels_array) = metadata_view.row_labels_array()? {
                println!(
                    "  Count: {}, Stride: {}",
                    row_labels_array.count(), row_labels_array.stride()
                );

                for i in 0..row_labels_array.count() {
                    if let Some(label) = metadata_view.row_label(i)? {
                        let label_str = std::str::from_utf8(label)
                            .unwrap_or("<invalid utf8>")
                            .trim_end_matches('\0');
                        println!("  Row {i}: {label_str}");
                    }
                }
            } else {
                println!("  No row labels found");
            }

            // Test column labels
            println!("Column labels:");
            if let Some(col_labels_array) = metadata_view.col_labels_array()? {
                println!(
                    "  Count: {}, Stride: {}",
                    col_labels_array.count(), col_labels_array.stride()
                );

                for i in 0..col_labels_array.count() {
                    if let Some(label) = metadata_view.col_label(i)? {
                        let label_str = std::str::from_utf8(label)
                            .unwrap_or("<invalid utf8>")
                            .trim_end_matches('\0');
                        println!("  Column {i}: {label_str}");
                    }
                }
            } else {
                println!("  No column labels found");
            }
        }
    } else {
        println!("No metadata found");
    }

    // Test direct label access through matrix
    if let Some(row_0_label) = matrix.row_label(0)? {
        let label_str = std::str::from_utf8(row_0_label)
            .unwrap_or("<invalid utf8>")
            .trim_end_matches('\0');
        println!("Row 0 label: {label_str}");
    }

    if let Some(col_1_label) = matrix.col_label(1)? {
        let label_str = std::str::from_utf8(col_1_label)
            .unwrap_or("<invalid utf8>")
            .trim_end_matches('\0');
        println!("Column 1 label: {label_str}");
    }

    // Clean up
    std::fs::remove_file(filename)
        .map_err(|_| binsparse_rs::Error::IoError("Failed to remove test file"))?;
    println!("Test completed successfully!");

    Ok(())
}
