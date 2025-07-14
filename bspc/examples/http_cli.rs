#[cfg(feature = "http")]
use clap::{Parser, Subcommand};

#[cfg(feature = "http")]
use bspc::{parse_range, HttpMatrix};

#[cfg(feature = "http")]
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(
    about = "BSPC HTTP CLI - Efficiently query remote binary sparse matrix files using HTTP range requests"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "http")]
#[derive(Subcommand)]
enum Commands {
    /// Query matrix data from remote URL
    Query {
        /// Remote URL to BSPC matrix file
        url: String,

        /// Row to query (optional)
        #[arg(long)]
        row: Option<usize>,

        /// Column to query (optional)
        #[arg(long)]
        col: Option<usize>,

        /// Row range (format: start:end)
        #[arg(long)]
        row_range: Option<String>,

        /// Column range (format: start:end)
        #[arg(long)]
        col_range: Option<String>,
    },
    /// Show remote matrix info
    Info {
        /// Remote URL to BSPC matrix file
        url: String,
    },
}

#[cfg(feature = "http")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let start_time = std::time::Instant::now();

    match &cli.command {
        Commands::Query {
            url,
            row,
            col,
            row_range,
            col_range,
        } => {
            handle_http_query(url, *row, *col, row_range, col_range).await?;
        }
        Commands::Info { url } => {
            handle_http_info(url).await?;
        }
    }

    let elapsed = start_time.elapsed();
    println!("Query completed in {elapsed:.2?}");

    Ok(())
}

#[cfg(not(feature = "http"))]
fn main() {
    eprintln!("This example requires the 'http' feature to be enabled.");
    eprintln!("Run with: cargo run --features http --example http_cli");
    std::process::exit(1);
}

#[cfg(feature = "http")]
async fn handle_http_query(
    url: &str,
    row: Option<usize>,
    col: Option<usize>,
    row_range: &Option<String>,
    col_range: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let matrix = HttpMatrix::new(url).await.map_err(|e| format!("{e:?}"))?;

    // Check if server supports range requests
    if !matrix
        .supports_range_requests()
        .await
        .map_err(|e| format!("{e:?}"))?
    {
        eprintln!("Warning: Server does not support range requests, performance may be poor");
    }

    if let (Some(r), Some(c)) = (row, col) {
        // Query specific element
        match matrix
            .get_element(r, c)
            .await
            .map_err(|e| format!("{e:?}"))?
        {
            Some(element) => println!("Element at ({r}, {c}): {element:?}"),
            None => println!("No element at ({r}, {c})"),
        }
    } else if let Some(row_range_str) = row_range {
        // Query row range
        let range = parse_range(row_range_str).map_err(|e| format!("{e:?}"))?;
        let elements = matrix
            .get_row_range(range.start, range.end)
            .await
            .map_err(|e| format!("{e:?}"))?;

        println!("Elements in row range {row_range_str}:");
        for (row, col, value) in elements {
            println!("  ({row}, {col}) = {value:?}");
        }
    } else if let (Some(r), Some(col_range_str)) = (row, col_range) {
        // Query specific row with column range
        let range = parse_range(col_range_str).map_err(|e| format!("{e:?}"))?;
        let elements = matrix
            .get_row_with_col_range(r, range.start, range.end)
            .await
            .map_err(|e| format!("{e:?}"))?;

        println!("Row {r} elements in column range {col_range_str}:");
        for (col, value) in elements {
            println!("  ({r}, {col}) = {value:?}");
        }
    } else if let Some(col_range_str) = col_range {
        // Query column range
        let range = parse_range(col_range_str).map_err(|e| format!("{e:?}"))?;
        let elements = matrix
            .get_col_range(range.start, range.end)
            .await
            .map_err(|e| format!("{e:?}"))?;

        println!("Elements in column range {col_range_str}:");
        for (row, col, value) in elements {
            println!("  ({row}, {col}) = {value:?}");
        }
    } else if let Some(r) = row {
        // Query specific row
        let elements = matrix.get_row(r).await.map_err(|e| format!("{e:?}"))?;
        println!("Row {r} elements:");
        for (col, value) in elements {
            println!("  ({r}, {col}) = {value:?}");
        }
    } else if let Some(c) = col {
        // Query specific column
        let elements = matrix.get_col(c).await.map_err(|e| format!("{e:?}"))?;
        println!("Column {c} elements:");
        for (row, value) in elements {
            println!("  ({row}, {c}) = {value:?}");
        }
    } else {
        // Show matrix info
        println!("Matrix dimensions: {} x {}", matrix.nrows(), matrix.ncols());
        println!("Non-zero elements: {}", matrix.nnz());
        println!("Format: {}", matrix.format());
        println!("Data type: {}", matrix.data_type());
        println!("Use --row and --col to query specific elements");
        println!("Use --row-range start:end to query row ranges");
        println!("Use --col-range start:end to query column ranges");
        println!("Use --row X --col-range start:end to query specific row with column range");
    }

    Ok(())
}

#[cfg(feature = "http")]
async fn handle_http_info(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let matrix = HttpMatrix::new(url).await.map_err(|e| format!("{e:?}"))?;

    println!("Remote Matrix Info:");
    println!("  URL: {url}");
    println!("  Dimensions: {} x {}", matrix.nrows(), matrix.ncols());
    println!("  Non-zero elements: {}", matrix.nnz());
    println!("  Format: {} ({})", matrix.format(), matrix.data_type());

    // Check server capabilities
    let supports_ranges = matrix
        .supports_range_requests()
        .await
        .map_err(|e| format!("{e:?}"))?;
    println!(
        "  Range requests: {}",
        if supports_ranges {
            "Supported"
        } else {
            "Not supported"
        }
    );

    if let Ok(size) = matrix.get_file_size().await {
        println!("  File size: {size} bytes");
    }

    Ok(())
}
