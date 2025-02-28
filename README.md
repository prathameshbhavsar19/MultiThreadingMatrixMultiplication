# Multithreading in Matrix Multiplication

## Project Overview
This project explores the impact of multithreading on the performance of **matrix multiplication** using C++ on a **MacBook Pro**. The goal is to compare execution times between **single-threaded** and **multi-threaded** implementations and analyze how threading improves computational efficiency.

## Features
- **Single-threaded matrix multiplication**
- **Multi-threaded matrix multiplication using std::thread**
- **Performance benchmarking and execution time comparison**
- **Scalability analysis with different matrix sizes and thread counts**

## Technologies Used
- **C++** (Standard Library for threading)
- **Xcode** (Development environment)
- **MacBook Pro** (Test machine)

## Prerequisites
- A C++ compiler (preferably Clang on macOS)
- Xcode or any C++ IDE

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/prathameshbhavsar19/multithreading-matrix.git
   cd multithreading-matrix
   ```
2. Compile the program:
   ```bash
   g++ -std=c++11 -pthread matrix_mult.cpp -o matrix_mult
   ```
3. Run the executable:
   ```bash
   ./matrix_mult
   ```

## Usage
1. Define matrix sizes and initialize random matrices.
2. Execute both **single-threaded** and **multi-threaded** matrix multiplication.
3. Observe the execution time for each approach.
4. Modify the number of threads and matrix size to analyze performance variations.

## Performance Evaluation
- Measure execution time using `std::chrono`.
- Compare the impact of increasing the number of threads.
- Analyze speedup and efficiency based on CPU core utilization.

## Sample Output
```
Matrix Size: 500x500
Single-threaded Execution Time: 3.45s
Multi-threaded Execution Time (4 threads): 1.12s
Speedup: ~3.08x
```

## Future Enhancements
- Implement **OpenMP** for easier parallelization.
- Optimize memory access to reduce cache misses.
- Extend to **GPU-based parallelization** using CUDA.

## Author
**Prathamesh Bhavsar**

## License
This project is open-source and available under the MIT License.

