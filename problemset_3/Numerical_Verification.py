import numpy as np
from scipy.integrate import nquad
# part a
def compute_integral(A, w):
    N = len(w)
    A_inv = np.linalg.inv(A)
    det_A_inv = np.linalg.det(A_inv)
    
    # Closed-form RHS
    rhs = np.sqrt((2 * np.pi)**N * det_A_inv) * np.exp(0.5 * w.T @ A_inv @ w)
    
    # Numerical LHS
    def integrand(*args):
        v = np.array(args)
        return np.exp(-0.5 * v @ A @ v + v @ w)
    
    ranges = [(-np.inf, np.inf) for _ in range(N)]
    lhs, _ = nquad(integrand, ranges)
    
    return lhs, rhs
#part b
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
A_prime = np.array([[4, 2, 1], [2, 1, 3], [1, 3, 6]])
w = np.array([1, 2, 3])

# For A
lhs_A, rhs_A = compute_integral(A, w)
print(f"Matrix A: LHS = {lhs_A}, RHS = {rhs_A}, Match = {np.isclose(lhs_A, rhs_A)}")

# For A'
try:
    lhs_A_prime, rhs_A_prime = compute_integral(A_prime, w)
    print(f"Matrix A': LHS = {lhs_A_prime}, RHS = {rhs_A_prime}, Match = {np.isclose(lhs_A_prime, rhs_A_prime)}")
except np.linalg.LinAlgError:
    print("Matrix A' is singular or not positive-definite.")
#part c
import numpy as np
from scipy.integrate import nquad

def compute_moment(A, w, powers):
    N = len(w)
    A_inv = np.linalg.inv(A)
    I = np.sqrt((2 * np.pi)**N * np.linalg.det(A_inv)) * np.exp(0.5 * w.T @ A_inv @ w)  # Closed-form I
    mu = A_inv @ w
    Sigma = A_inv
    
    # Define integrand
    def integrand(*args):
        v = np.array(args)
        product = np.prod([v[i]**p for i, p in enumerate(powers)])
        return product * np.exp(-0.5 * v @ A @ v + v @ w)
    
    ranges = [(-np.inf, np.inf) for _ in range(N)]
    numerical, _ = nquad(integrand, ranges)
    
    # Closed-form calculation
    total_power = sum(powers)
    non_zero_indices = np.where(np.array(powers) > 0)[0]
    non_zero_powers = [powers[i] for i in non_zero_indices]
    
    if total_power == 1:
        i = non_zero_indices[0]
        closed = I * mu[i]
    elif total_power == 2:
        if len(non_zero_indices) == 1:  # e.g., [2,0,0]
            i = non_zero_indices[0]
            closed = I * (mu[i]**2 + Sigma[i, i])
        else:  # e.g., [1,1,0]
            i, j = non_zero_indices
            closed = I * (mu[i] * mu[j] + Sigma[i, j])
    elif total_power == 3:
        if len(non_zero_indices) == 2:  # e.g., [2,1,0]
            i, j = non_zero_indices
            pi, pj = non_zero_powers
            if pi == 2 and pj == 1:
                term = mu[i]**2 * mu[j] + 2 * mu[i] * Sigma[i, j] + Sigma[i, i] * mu[j]
            elif pi == 1 and pj == 2:
                term = mu[i] * mu[j]**2 + 2 * mu[j] * Sigma[i, j] + Sigma[j, j] * mu[i]
            else:
                raise ValueError("Unsupported powers for third moment")
            closed = I * term
        else:
            raise ValueError("Third moment requires two non-zero indices")
    elif total_power == 4:
        if len(non_zero_indices) == 2 and non_zero_powers == [2, 2]:  # e.g., [2,2,0]
            i, j = non_zero_indices
            term = (mu[i]**2 * mu[j]**2 + 
                    2 * mu[i]**2 * Sigma[j, j] + 
                    2 * mu[j]**2 * Sigma[i, i] + 
                    4 * mu[i] * mu[j] * Sigma[i, j] + 
                    Sigma[i, i] * Sigma[j, j] + 
                    2 * Sigma[i, j]**2)
            closed = I * term
        else:
            raise ValueError("Fourth moment requires two indices with power 2")
    else:
        raise NotImplementedError("Higher moments not implemented")
    
    return numerical, closed
#part c about verification for specific moments
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
w = np.array([1, 2, 3])

# First moments
v1_num, v1_closed = compute_moment(A, w, [1, 0, 0])
v2_num, v2_closed = compute_moment(A, w, [0, 1, 0])
v3_num, v3_closed = compute_moment(A, w, [0, 0, 1])

# Second moments
v1v2_num, v1v2_closed = compute_moment(A, w, [1, 1, 0])
v2v3_num, v2v3_closed = compute_moment(A, w, [0, 1, 1])
v1v3_num, v1v3_closed = compute_moment(A, w, [1, 0, 1])

# Third moments
v1_sq_v2_num, v1_sq_v2_closed = compute_moment(A, w, [2, 1, 0])
v2_v3_sq_num, v2_v3_sq_closed = compute_moment(A, w, [0, 1, 2])

# Fourth moments
v1_sq_v2_sq_num, v1_sq_v2_sq_closed = compute_moment(A, w, [2, 2, 0])
v2_sq_v3_sq_num, v2_sq_v3_sq_closed = compute_moment(A, w, [0, 2, 2])

# Print results (example for first moment)
print(f"⟨v₁⟩: Numerical = {v1_num:.4f}, Closed = {v1_closed:.4f}, Match = {np.isclose(v1_num, v1_closed, rtol=1e-3)}")
