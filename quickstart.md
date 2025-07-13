# Quickstart Guide for BSPC

`bspc` can be used to handled large sparse matrices efficiently. Below is a quickstart guide to get you started with writing and reading sparse matrices using the BSPC format.


### Writing a Sparse Matrix

To write a sparse matrix to a BSPC file, you can use the following command:

```bash
cargo run --example write_matrix
```

```txt
Writing large sparse matrix using streaming approach...
Matrix dimensions: 100_000_000 x 2_000_000
Sparse rows: 8 rows with 1_000_000 elements each
Total non-zeros: 8_000_000
Built sparse elements in 94.544334ms
Written matrix 100_000_000 x 2_000_000 with 8_000_000 non-zeros to example_matrix.bspc
Matrix + bloom filters written in 26.992572458s

Run 'cargo run --example read_matrix' to read it back!
```

You can check the size of the generated BSPC file:

```bash
ls -lh example_matrix.bspc
```

```txt
122M Jul 13 19:09 example_matrix.bspc
```

### Reading a Sparse Matrix

To read the matrix back from the BSPC file, you can use the following command:

```bash
cargo run --example read_matrix
```

```txt
Reading sparse matrix from 'example_matrix.bspc'...
Opening file and parsing header...
Header parsed in 0.177ms
Accessing matrix metadata...
Metadata accessed in 0.000ms

Matrix Information:
   Dimensions: 100000000 x 2000000
   Non-zeros: 8000000
   Format: COO
   Data type: f64
   Sparsity: 0.000004%

Testing element access performance:
Testing likely positions (based on generation pattern):
   Access 1: matrix[0, 0] → 0.0 (MISS in 72.757ms)
   Access 2: matrix[7, 0] → 0.0 (MISS in 51.429ms)
   Access 3: matrix[14, 0] → 0.0 (MISS in 51.210ms)
   Access 4: matrix[21, 0] → 0.0 (MISS in 51.014ms)

Testing random positions (likely sparse):
   Random 1: matrix[500, 500] → 550.0 (HIT in 19.308ms)
   Random 2: matrix[90000, 500] → 90050.0 (HIT in 45.895ms)
   Random 3: matrix[50000, 250] → 50025.0 (HIT in 39.271ms)

Lazy access to first 10 values of row 10:
   Row 90000: 1000000 non-zero elements (view created in 72.349ms)
   First 10 column values:
     [0] = 90000 (in 0.002ms)
     [1] = 90000.1 (in 0.000ms)
     [2] = 90000.2 (in 0.000ms)
     [3] = 90000.3 (in 0.000ms)
     [4] = 90000.4 (in 0.000ms)
     [5] = 90000.5 (in 0.000ms)
     [6] = 90000.6 (in 0.000ms)
     [7] = 90000.7 (in 0.000ms)
     [8] = 90000.8 (in 0.000ms)
     [9] = 90000.9 (in 0.000ms)

Total operation time: 403.653ms
```

woo! we did multiple reads in under 500ms!