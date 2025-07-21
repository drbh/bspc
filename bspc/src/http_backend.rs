//! HTTP backend for remote BSPC file access
//!
//! This module provides HTTP-based access to remote BSPC files using range requests
//! for efficient partial downloads. Only available when the "http" feature is enabled.

#[cfg(feature = "http")]
pub mod http_impl {
    use binsparse_rs::{array::ArrayValue, Error, Result};
    use bspc_core::{BspcHeader, DataType, MatrixFormat};
    use reqwest::Client;
    use std::collections::HashMap;
    use std::ops::Range;
    use tokio::sync::RwLock;

    /// HTTP client for efficient range-based access to remote BSPC files
    pub struct HttpMatrix {
        client: Client,
        url: String,
        header: BspcHeader,
        cache: RwLock<HashMap<Range<usize>, Vec<u8>>>,
    }

    impl HttpMatrix {
        /// Create a new HTTP matrix client
        pub async fn new(url: &str) -> Result<Self> {
            let client = Client::new();

            // Read and validate header using range request
            let header_bytes = Self::fetch_range(&client, url, 0, BspcHeader::SIZE).await?;

            let header = BspcHeader::from_bytes(&header_bytes)
                .map_err(|_| Error::InvalidState("Invalid BSPC header format"))?;

            Ok(Self {
                client,
                url: url.to_string(),
                header,
                cache: RwLock::new(HashMap::new()),
            })
        }

        /// Fetch a byte range from the remote file
        async fn fetch_range(
            client: &Client,
            url: &str,
            start: usize,
            len: usize,
        ) -> Result<Vec<u8>> {
            let end = start + len - 1;
            let range_header = format!("bytes={start}-{end}");

            let response = client
                .get(url)
                .header("Range", range_header)
                .send()
                .await
                .map_err(|_| Error::IoError("Failed to fetch range"))?;

            if !response.status().is_success() && response.status().as_u16() != 206 {
                return Err(Error::IoError("HTTP request failed"));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|_| Error::IoError("Failed to read response bytes"))?;

            Ok(bytes.to_vec())
        }

        /// Get cached data or fetch from server
        async fn get_cached_range(&self, range: Range<usize>) -> Result<Vec<u8>> {
            {
                let cache = self.cache.read().await;
                if let Some(data) = cache.get(&range) {
                    return Ok(data.clone());
                }
            }

            let data = Self::fetch_range(&self.client, &self.url, range.start, range.len()).await?;

            {
                let mut cache = self.cache.write().await;
                cache.insert(range, data.clone());
            }

            Ok(data)
        }

        /// Get matrix dimensions
        pub fn nrows(&self) -> usize {
            self.header.nrows as usize
        }

        pub fn ncols(&self) -> usize {
            self.header.ncols as usize
        }

        pub fn nnz(&self) -> usize {
            self.header.nnz as usize
        }

        /// Get matrix format
        pub fn format(&self) -> MatrixFormat {
            MatrixFormat::from_u8(self.header.format_type).unwrap_or(MatrixFormat::Coo)
        }

        /// Get data type
        pub fn data_type(&self) -> DataType {
            DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64)
        }

        /// Get a specific element with efficient range queries
        pub async fn get_element(&self, row: usize, col: usize) -> Result<Option<ArrayValue>> {
            if row >= self.nrows() || col >= self.ncols() {
                return Err(Error::InvalidState("Index out of bounds"));
            }

            // For now, implement a simple COO search - in production you'd optimize this
            // by reading chunks and building indices
            let values_range = self.header.values_offset as usize
                ..self.header.values_offset as usize + self.header.values_size as usize;
            let row_indices_range = self.header.indices_0_offset as usize
                ..self.header.indices_0_offset as usize + self.header.indices_0_size as usize;
            let col_indices_range = self.header.indices_1_offset as usize
                ..self.header.indices_1_offset as usize + self.header.indices_1_size as usize;

            let (values_bytes, row_indices_bytes, col_indices_bytes) = tokio::try_join!(
                self.get_cached_range(values_range),
                self.get_cached_range(row_indices_range),
                self.get_cached_range(col_indices_range)
            )?;

            let data_type = DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64);
            let nnz = self.header.nnz as usize;

            // Convert row and column indices to u32 slices
            let row_indices = self.bytes_to_u32_slice(&row_indices_bytes)?;
            let col_indices = self.bytes_to_u32_slice(&col_indices_bytes)?;

            // Search for the element
            for i in 0..nnz {
                if row_indices[i] as usize == row && col_indices[i] as usize == col {
                    return Ok(Some(self.extract_value_at_index(
                        &values_bytes,
                        i,
                        data_type,
                    )?));
                }
            }

            Ok(None)
        }

        /// Get a range of rows efficiently
        pub async fn get_row_range(
            &self,
            start_row: usize,
            end_row: usize,
        ) -> Result<Vec<(usize, usize, ArrayValue)>> {
            if start_row >= self.nrows() || end_row > self.nrows() || start_row >= end_row {
                return Err(Error::InvalidState("Invalid row range"));
            }

            // Fetch all required data in parallel
            let values_range = self.header.values_offset as usize
                ..self.header.values_offset as usize + self.header.values_size as usize;
            let row_indices_range = self.header.indices_0_offset as usize
                ..self.header.indices_0_offset as usize + self.header.indices_0_size as usize;
            let col_indices_range = self.header.indices_1_offset as usize
                ..self.header.indices_1_offset as usize + self.header.indices_1_size as usize;

            let (values_bytes, row_indices_bytes, col_indices_bytes) = tokio::try_join!(
                self.get_cached_range(values_range),
                self.get_cached_range(row_indices_range),
                self.get_cached_range(col_indices_range)
            )?;

            let data_type = DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64);
            let nnz = self.header.nnz as usize;

            // Convert indices to u32 slices
            let row_indices = self.bytes_to_u32_slice(&row_indices_bytes)?;
            let col_indices = self.bytes_to_u32_slice(&col_indices_bytes)?;

            let mut results = Vec::new();

            // Filter elements in the row range
            for i in 0..nnz {
                let file_row = row_indices[i] as usize;
                let file_col = col_indices[i] as usize;

                if file_row >= start_row && file_row < end_row && file_col < self.ncols() {
                    let value = self.extract_value_at_index(&values_bytes, i, data_type)?;
                    results.push((file_row, file_col, value));
                }
            }

            Ok(results)
        }

        /// Get a specific row
        pub async fn get_row(&self, row: usize) -> Result<Vec<(usize, ArrayValue)>> {
            let elements = self.get_row_range(row, row + 1).await?;
            Ok(elements
                .into_iter()
                .map(|(_, col, value)| (col, value))
                .collect())
        }

        /// Get a specific row with column range filter
        pub async fn get_row_with_col_range(
            &self,
            row: usize,
            start_col: usize,
            end_col: usize,
        ) -> Result<Vec<(usize, ArrayValue)>> {
            if row >= self.nrows() {
                return Err(Error::InvalidState("Row index out of bounds"));
            }
            if start_col >= self.ncols() || end_col > self.ncols() || start_col >= end_col {
                return Err(Error::InvalidState("Invalid column range"));
            }

            // Fetch all required data in parallel
            let values_range = self.header.values_offset as usize
                ..self.header.values_offset as usize + self.header.values_size as usize;
            let row_indices_range = self.header.indices_0_offset as usize
                ..self.header.indices_0_offset as usize + self.header.indices_0_size as usize;
            let col_indices_range = self.header.indices_1_offset as usize
                ..self.header.indices_1_offset as usize + self.header.indices_1_size as usize;

            let (values_bytes, row_indices_bytes, col_indices_bytes) = tokio::try_join!(
                self.get_cached_range(values_range),
                self.get_cached_range(row_indices_range),
                self.get_cached_range(col_indices_range)
            )?;

            let data_type = DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64);
            let nnz = self.header.nnz as usize;

            // Convert indices to u32 slices
            let row_indices = self.bytes_to_u32_slice(&row_indices_bytes)?;
            let col_indices = self.bytes_to_u32_slice(&col_indices_bytes)?;

            let mut results = Vec::new();

            // Filter elements in the specific row and column range
            for i in 0..nnz {
                let file_row = row_indices[i] as usize;
                let file_col = col_indices[i] as usize;

                if file_row == row && file_col >= start_col && file_col < end_col {
                    let value = self.extract_value_at_index(&values_bytes, i, data_type)?;
                    results.push((file_col, value));
                }
            }

            Ok(results)
        }

        /// Get a column range efficiently
        pub async fn get_col_range(
            &self,
            start_col: usize,
            end_col: usize,
        ) -> Result<Vec<(usize, usize, ArrayValue)>> {
            if start_col >= self.ncols() || end_col > self.ncols() || start_col >= end_col {
                return Err(Error::InvalidState("Invalid column range"));
            }

            // Fetch all required data in parallel
            let values_range = self.header.values_offset as usize
                ..self.header.values_offset as usize + self.header.values_size as usize;
            let row_indices_range = self.header.indices_0_offset as usize
                ..self.header.indices_0_offset as usize + self.header.indices_0_size as usize;
            let col_indices_range = self.header.indices_1_offset as usize
                ..self.header.indices_1_offset as usize + self.header.indices_1_size as usize;

            let (values_bytes, row_indices_bytes, col_indices_bytes) = tokio::try_join!(
                self.get_cached_range(values_range),
                self.get_cached_range(row_indices_range),
                self.get_cached_range(col_indices_range)
            )?;

            let data_type = DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64);
            let nnz = self.header.nnz as usize;

            // Convert indices to u32 slices
            let row_indices = self.bytes_to_u32_slice(&row_indices_bytes)?;
            let col_indices = self.bytes_to_u32_slice(&col_indices_bytes)?;

            let mut results = Vec::new();

            // Filter elements in the column range
            for i in 0..nnz {
                let file_row = row_indices[i] as usize;
                let file_col = col_indices[i] as usize;

                if file_col >= start_col && file_col < end_col && file_row < self.nrows() {
                    let value = self.extract_value_at_index(&values_bytes, i, data_type)?;
                    results.push((file_row, file_col, value));
                }
            }

            Ok(results)
        }

        /// Get a specific column
        pub async fn get_col(&self, col: usize) -> Result<Vec<(usize, ArrayValue)>> {
            if col >= self.ncols() {
                return Err(Error::InvalidState("Column index out of bounds"));
            }

            // Fetch all required data
            let values_range = self.header.values_offset as usize
                ..self.header.values_offset as usize + self.header.values_size as usize;
            let row_indices_range = self.header.indices_0_offset as usize
                ..self.header.indices_0_offset as usize + self.header.indices_0_size as usize;
            let col_indices_range = self.header.indices_1_offset as usize
                ..self.header.indices_1_offset as usize + self.header.indices_1_size as usize;

            let (values_bytes, row_indices_bytes, col_indices_bytes) = tokio::try_join!(
                self.get_cached_range(values_range),
                self.get_cached_range(row_indices_range),
                self.get_cached_range(col_indices_range)
            )?;

            let data_type = DataType::from_u8(self.header.data_type).unwrap_or(DataType::F64);
            let nnz = self.header.nnz as usize;

            // Convert indices to u32 slices
            let row_indices = self.bytes_to_u32_slice(&row_indices_bytes)?;
            let col_indices = self.bytes_to_u32_slice(&col_indices_bytes)?;

            let mut results = Vec::new();

            // Filter elements in the column
            for i in 0..nnz {
                let file_row = row_indices[i] as usize;
                let file_col = col_indices[i] as usize;

                if file_col == col && file_row < self.nrows() {
                    let value = self.extract_value_at_index(&values_bytes, i, data_type)?;
                    results.push((file_row, value));
                }
            }

            Ok(results)
        }

        /// Convert bytes to u32 slice
        fn bytes_to_u32_slice(&self, bytes: &[u8]) -> Result<Vec<u32>> {
            if bytes.len() % 4 != 0 {
                return Err(Error::InvalidState("Invalid u32 array size"));
            }

            let mut result = Vec::new();
            for chunk in bytes.chunks_exact(4) {
                let array: [u8; 4] = chunk.try_into().unwrap();
                result.push(u32::from_le_bytes(array));
            }
            Ok(result)
        }

        /// Extract value at specific index based on data type
        fn extract_value_at_index(
            &self,
            bytes: &[u8],
            index: usize,
            data_type: DataType,
        ) -> Result<ArrayValue> {
            let element_size = data_type.size_bytes();
            let start = index * element_size;
            let end = start + element_size;

            if end > bytes.len() {
                return Err(Error::InvalidState("Index out of bounds"));
            }

            let value_bytes = &bytes[start..end];

            match data_type {
                DataType::F32 => {
                    let array: [u8; 4] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::Float32(f32::from_le_bytes(array)))
                }
                DataType::F64 => {
                    let array: [u8; 8] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::Float64(f64::from_le_bytes(array)))
                }
                DataType::I32 => {
                    let array: [u8; 4] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::Int32(i32::from_le_bytes(array)))
                }
                DataType::I64 => {
                    let array: [u8; 8] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::Int64(i64::from_le_bytes(array)))
                }
                DataType::U32 => {
                    let array: [u8; 4] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::UInt32(u32::from_le_bytes(array)))
                }
                DataType::U64 => {
                    let array: [u8; 8] = value_bytes.try_into().unwrap();
                    Ok(ArrayValue::UInt64(u64::from_le_bytes(array)))
                }
            }
        }

        /// Get file size from server
        pub async fn get_file_size(&self) -> Result<u64> {
            let response = self
                .client
                .head(&self.url)
                .send()
                .await
                .map_err(|_| Error::IoError("Failed to get file info"))?;

            if let Some(content_length) = response.headers().get("content-length") {
                let size_str = content_length
                    .to_str()
                    .map_err(|_| Error::InvalidState("Invalid content-length header"))?;
                let size = size_str
                    .parse::<u64>()
                    .map_err(|_| Error::InvalidState("Invalid content-length value"))?;
                Ok(size)
            } else {
                Err(Error::InvalidState("No content-length header"))
            }
        }

        /// Check if the server supports range requests
        pub async fn supports_range_requests(&self) -> Result<bool> {
            let response = self
                .client
                .head(&self.url)
                .send()
                .await
                .map_err(|_| Error::IoError("Failed to check range support"))?;

            Ok(response
                .headers()
                .get("accept-ranges")
                .is_some_and(|v| v.as_bytes() == b"bytes"))
        }
    }

    /// Parse range string (e.g., "10:20") to range
    pub fn parse_range(range_str: &str) -> Result<std::ops::Range<usize>> {
        let parts: Vec<&str> = range_str.split(':').collect();
        if parts.len() != 2 {
            return Err(Error::InvalidState("Invalid range format, use start:end"));
        }

        let start = parts[0]
            .parse::<usize>()
            .map_err(|_| Error::InvalidState("Invalid start index"))?;
        let end = parts[1]
            .parse::<usize>()
            .map_err(|_| Error::InvalidState("Invalid end index"))?;

        if start >= end {
            return Err(Error::InvalidState("Start must be less than end"));
        }

        Ok(start..end)
    }
}

#[cfg(feature = "http")]
pub use http_impl::*;

#[cfg(not(feature = "http"))]
pub mod http_stub {
    use binsparse_rs::{Error, Result};

    pub struct HttpMatrix;

    impl HttpMatrix {
        pub async fn new(_url: &str) -> Result<Self> {
            Err(Error::InvalidState(
                "HTTP support not enabled. Build with --features http",
            ))
        }
    }

    pub fn parse_range(_range_str: &str) -> Result<std::ops::Range<usize>> {
        Err(Error::InvalidState(
            "HTTP support not enabled. Build with --features http",
        ))
    }
}

#[cfg(not(feature = "http"))]
pub use http_stub::*;
