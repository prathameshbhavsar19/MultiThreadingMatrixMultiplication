import numpy as np
import threading
import time

# Function to read a matrix from a file
def read_matrix(filename, rows, cols):
    try:
        with open(filename, "rb") as f:
            data = np.fromfile(f, dtype=np.float64)
            if data.size != rows * cols:
                raise ValueError(f"File {filename} does not contain enough data to form a {rows}x{cols} matrix")
            return data.reshape(rows, cols)
    except Exception as e:
        raise ValueError(f"Error reading file {filename}: {e}")

# Single-threaded matrix multiplication
def matmul_single_threaded(mat1, mat2):
    return np.dot(mat1, mat2)

# Threaded worker function for computing a subset of rows
def threaded_matmul_worker(mat1, mat2, result, row_start, row_end):
    for i in range(row_start, row_end):
        result[i] = np.dot(mat1[i], mat2)

# Multi-threaded matrix multiplication
def matmul_multithreaded(mat1, mat2, num_threads):
    rows, cols = mat1.shape[0], mat2.shape[1]
    result = np.zeros((rows, cols))
    
    chunk_size = rows // num_threads
    threads = []
    
    for i in range(num_threads):
        row_start = i * chunk_size
        row_end = rows if i == num_threads - 1 else (i + 1) * chunk_size
        thread = threading.Thread(target=threaded_matmul_worker, args=(mat1, mat2, result, row_start, row_end))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return result

# Part 4: Compare single-threaded and multi-threaded implementations
def part4():
    rows, cols = 1000, 1000  # Dimensions of the matrices
    num_threads = 4  # Default thread count for multi-threading

    # Read matrices
    mat1 = read_matrix("data1", rows, cols)
    mat2 = read_matrix("data2", rows, cols)
    print(f"Shape of mat1: {mat1.shape}, Shape of mat2: {mat2.shape}")  # Debugging

    # Single-threaded performance
    start_time = time.time()
    mat3_single = matmul_single_threaded(mat1, mat2)
    end_time = time.time()
    single_thread_time = end_time - start_time
    print(f"Single-threaded Execution Time: {single_thread_time:.2f} seconds")

    # Multi-threaded performance
    start_time = time.time()
    mat3_multi = matmul_multithreaded(mat1, mat2, num_threads)
    end_time = time.time()
    multi_thread_time = end_time - start_time
    print(f"Multi-threaded Execution Time with {num_threads} threads: {multi_thread_time:.2f} seconds")

    # Verify correctness
    if np.allclose(mat3_single, mat3_multi):
        print("The results of single-threaded and multi-threaded implementations match!")
    else:
        print("The results do NOT match. Please check your implementation.")

    # Performance comparison
    print("\nPerformance Comparison:")
    print(f"Single-threaded: {single_thread_time:.2f} seconds")
    print(f"Multi-threaded ({num_threads} threads): {multi_thread_time:.2f} seconds")
    print(f"Speedup: {single_thread_time / multi_thread_time:.2f}x")

# Entry point for Part 4
if __name__ == "__main__":
    start_time = time.time()
    try:
        part4()
    except Exception as e:
        print(f"An error occurred: {e}")
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
