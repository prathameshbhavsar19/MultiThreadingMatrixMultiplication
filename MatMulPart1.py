import numpy as np
import time

# Function to read a matrix from a file
def read_matrix(filename, rows, cols):
    try:
        with open(filename, "rb") as f:  # Open in binary mode to handle raw data
            data = np.fromfile(f, dtype=np.float64)
            if data.size != rows * cols:
                raise ValueError(f"File {filename} does not contain enough data to form a {rows}x{cols} matrix")
            return data.reshape(rows, cols)
    except Exception as e:
        raise ValueError(f"Error reading file {filename}: {e}")

# Function to write a matrix to a file
def write_matrix(filename, matrix):
    if len(matrix.shape) != 2:
        raise ValueError(f"Expected a 2D array, but got {len(matrix.shape)}D array")
    matrix.tofile(filename)  # Save as binary file for consistency

# Function to perform matrix multiplication
def matmul(mat1, mat2):
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("Matrix dimensions do not match for multiplication")
    return np.dot(mat1, mat2)

# Part 1: Read matrices, multiply, and write output
def part1():
    rows, cols = 1000, 1000  # Dimensions of the matrices
    mat1 = read_matrix("data1", rows, cols)
    mat2 = read_matrix("data2", rows, cols)
    print(f"Shape of mat1: {mat1.shape}, Shape of mat2: {mat2.shape}")  # Debugging
    mat3 = matmul(mat1, mat2)
    print(f"Shape of mat3: {mat3.shape}")  # Debugging
    write_matrix("data3", mat3)
    print("Matrix multiplication (Part 1) completed successfully!")

# Entry point of the program
if __name__ == "__main__":
    start_time = time.time()
    try:
        part1()
    except Exception as e:
        print(f"An error occurred: {e}")
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
