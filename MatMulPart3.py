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

# Threaded worker function for computing a subset of rows
def threaded_matmul_worker(mat1, mat2, result, row_start, row_end):
    for i in range(row_start, row_end):
        result[i] = np.dot(mat1[i], mat2)

# Function to perform multi-threaded matrix multiplication
def matmul_multithreaded(mat1, mat2, num_threads):
    rows, cols = mat1.shape[0], mat2.shape[1]
    result = np.zeros((rows, cols))
    
    # Divide rows among threads
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

# Part 3: Measure performance for varying thread counts
def part3():
    rows, cols = 1000, 1000  # Dimensions of the matrices
    thread_counts = [1, 2, 4, 8]  # Different numbers of threads to test

    # Read matrices
    mat1 = read_matrix("data1", rows, cols)
    mat2 = read_matrix("data2", rows, cols)
    print(f"Shape of mat1: {mat1.shape}, Shape of mat2: {mat2.shape}")  # Debugging

    # Measure performance
    results = {}
    for num_threads in thread_counts:
        start_time = time.time()
        mat3 = matmul_multithreaded(mat1, mat2, num_threads)
        end_time = time.time()
        execution_time = end_time - start_time
        results[num_threads] = execution_time
        print(f"Threads: {num_threads}, Execution Time: {execution_time:.2f} seconds")
    
    # Display performance results
    print("\nPerformance Summary:")
    for num_threads, exec_time in results.items():
        print(f"{num_threads} thread(s): {exec_time:.2f} seconds")

# Entry point for Part 3
if __name__ == "__main__":
    start_time = time.time()
    try:
        part3()
    except Exception as e:
        print(f"An error occurred: {e}")
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
