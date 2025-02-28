import numpy as np
import threading
import time
import matplotlib.pyplot as plt

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

# Generate random matrices
def generate_matrices(size):
    mat1 = np.random.rand(size, size)
    mat2 = np.random.rand(size, size)
    return mat1, mat2

# Part 5: Performance analysis
def part5():
    sizes = [100, 200, 500, 1000]  # Matrix sizes to test
    thread_counts = [1, 2, 4, 8]   # Thread counts to test

    results = []

    for size in sizes:
        mat1, mat2 = generate_matrices(size)
        print(f"\nMatrix size: {size}x{size}")
        size_results = {"size": size, "times": {}}
        
        for num_threads in thread_counts:
            start_time = time.time()
            if num_threads == 1:
                # Single-threaded
                matmul_single_threaded(mat1, mat2)
            else:
                # Multi-threaded
                matmul_multithreaded(mat1, mat2, num_threads)
            end_time = time.time()
            exec_time = end_time - start_time
            size_results["times"][num_threads] = exec_time
            print(f"Threads: {num_threads}, Time: {exec_time:.4f} seconds")
        
        results.append(size_results)

    # Plot results
    for result in results:
        size = result["size"]
        times = result["times"]
        x = list(times.keys())
        y = list(times.values())
        plt.plot(x, y, marker='o', label=f"Size: {size}x{size}")

    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Performance Analysis of Matrix Multiplication")
    plt.legend()
    plt.grid()
    plt.show()

# Entry point for Part 5
if __name__ == "__main__":
    start_time = time.time()
    try:
        part5()
    except Exception as e:
        print(f"An error occurred: {e}")
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
