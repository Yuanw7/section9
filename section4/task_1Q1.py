import numpy as np
import time
import matplotlib.pyplot as plt

def matrix_add(A, B):
    """Add two matrices."""
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]

def matrix_subtract(A, B):
    """Subtract two matrices."""
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]

def split_matrix(M):
    """Split a matrix into four submatrices."""
    n = len(M)
    mid = n // 2
    M11 = [row[:mid] for row in M[:mid]]
    M12 = [row[mid:] for row in M[:mid]]
    M21 = [row[:mid] for row in M[mid:]]
    M22 = [row[mid:] for row in M[mid:]]
    return M11, M12, M21, M22

def naive_dc_mult(A, B):
    """Naive divide-and-conquer matrix multiplication."""
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Split matrices into submatrices
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Compute 8 recursive multiplications
    C11 = matrix_add(naive_dc_mult(A11, B11), naive_dc_mult(A12, B21))
    C12 = matrix_add(naive_dc_mult(A11, B12), naive_dc_mult(A12, B22))
    C21 = matrix_add(naive_dc_mult(A21, B11), naive_dc_mult(A22, B21))
    C22 = matrix_add(naive_dc_mult(A21, B12), naive_dc_mult(A22, B22))
    
    # Combine results
    top = [C11[i] + C12[i] for i in range(n//2)]
    bottom = [C21[i] + C22[i] for i in range(n//2)]
    return top + bottom

# Example usage:
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = naive_dc_mult(A, B)
print("Result of 2x2 multiplication:", C)  # Output: [[19, 22], [43, 50]]

def time_measurement(max_n=16):
    sizes = [2**i for i in range(1, int(np.log2(max_n)) + 1)]
    times = []
    for n in sizes:
        A = np.random.rand(n, n).tolist()
        B = np.random.rand(n, n).tolist()
        start = time.time()
        naive_dc_mult(A, B)
        end = time.time()
        times.append(end - start)
    return sizes, times

sizes, times = time_measurement()

# Plot log-log
log_sizes = np.log2(sizes)
log_times = np.log2(times)

plt.figure(figsize=(10, 6))
plt.scatter(log_sizes, log_times, label='Empirical (Naive DC)')
plt.plot(log_sizes, 3 * log_sizes + np.log2(times[0]) - 3 * log_sizes[0], '--', label='Theoretical O(n³)')
plt.xlabel('log2(n)')
plt.ylabel('log2(time)')
plt.legend()
plt.title('Time Complexity Comparison')
plt.show()
