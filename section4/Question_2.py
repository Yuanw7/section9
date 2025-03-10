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

def combine_matrices(C11, C12, C21, C22):
    """Combine four submatrices into a single matrix."""
    n = len(C11) * 2
    C = [[0 for _ in range(n)] for _ in range(n)]
    mid = n // 2
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    return C

def strassen_mult(A, B):
    """Strassen's algorithm for matrix multiplication."""
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Split matrices into submatrices
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Compute intermediate matrices M1-M7
    M1 = strassen_mult(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen_mult(matrix_add(A21, A22), B11)
    M3 = strassen_mult(A11, matrix_subtract(B12, B22))
    M4 = strassen_mult(A22, matrix_subtract(B21, B11))
    M5 = strassen_mult(matrix_add(A11, A12), B22)
    M6 = strassen_mult(matrix_subtract(A21, A11), matrix_add(B11, B12))
    M7 = strassen_mult(matrix_subtract(A12, A22), matrix_add(B21, B22))
    
    # Compute C submatrices
    C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_subtract(matrix_add(M1, M3), M2), M6)
    
    return combine_matrices(C11, C12, C21, C22)

# Example usage:
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = strassen_mult(A, B)
print("Strassen result (2x2):", C)  # Output: [[19, 22], [43, 50]]

def measure_strassen_time(max_n=64):
    sizes = [2**i for i in range(1, int(np.log2(max_n)) + 1)]
    times = []
    for n in sizes:
        A = np.random.rand(n, n).tolist()
        B = np.random.rand(n, n).tolist()
        start = time.time()
        strassen_mult(A, B)
        end = time.time()
        times.append(end - start)
    return sizes, times

sizes, times = measure_strassen_time()

# Compute logarithms for plotting
log_sizes = np.log2(sizes)
log_times = np.log2(times)

# Linear regression to estimate slope (critical exponent)
slope, intercept = np.polyfit(log_sizes, log_times, 1)
estimated_exponent = slope

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(log_sizes, log_times, label='Empirical (Strassen)')
plt.plot(log_sizes, log2(7) * log_sizes + (log_times[0] - log2(7) * log_sizes[0]), '--', label=f'Theoretical $O(n^{{\log_2 7}})$')
plt.xlabel('log2(n)')
plt.ylabel('log2(time)')
plt.legend()
plt.title(f'Strassen Critical Exponent: Empirical ≈ {estimated_exponent:.3f} vs Theoretical ≈ {np.log2(7):.3f}')
plt.show()
