# part 1
import numpy as np
from scipy import integrate
from scipy.special import gamma

# =============================================
# Task 1: Density of States and Partition Function
# =============================================

def density_of_states(E, m, omega, d=2):
    """1a: Density of states for harmonic oscillator (2D linear case)"""
    volume = (2 * np.pi * m * omega * E)**(d/2) / gamma(d/2 + 1)
    return np.gradient(volume, E)  # g(E) = dΩ/dE

def partition_function(beta, m, omega):
    """1b: Partition function via integration"""
    integrand = lambda E: density_of_states(E, m, omega) * np.exp(-beta * E)
    return integrate.quad(integrand, 0, np.inf)[0]

def nonlinear_density_of_states(E, m, omega, lambda_coeff):
    """1c: Density of states for nonlinear oscillator (approx)"""
    # Momentum integral first (p^2 term dominates for small λ)
    p_integral = np.pi * np.sqrt(2 * m * E)  # Approximate solution
    return p_integral / (np.sqrt(2 * m * (E - 0.5 * m * omega**2 * E**2 / (1 + lambda_coeff * E))))

# Example usage:
m, omega, lambda_val = 1.0, 1.0, 0.1
print(f"Z(β=1) = {partition_function(1, m, omega):.4f}")

#part2
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# =============================================
# Task 2: Double Pendulum Analysis
# =============================================

# --------------------------
# 2a: Lagrangian and EOMs
# --------------------------
def lagrangian_equations(t, state, L1, L2, m1, m2, g):
    theta1, theta2, omega1, omega2 = state
    
    # Equations of motion derived via Euler-Lagrange
    delta = theta2 - theta1
    denom = m1 + m2 * np.sin(delta)**2
    
    # Matrix form coefficients
    M = np.array([
        [(m1 + m2)*L1, m2*L2*np.cos(delta)],
        [m2*L1*np.cos(delta), m2*L2]
    ])
    
    C = np.array([
        -m2*L2*omega2**2*np.sin(delta) - (m1 + m2)*g*np.sin(theta1),
        m2*L1*omega1**2*np.sin(delta) - m2*g*np.sin(theta2)
    ])
    
    accelerations = np.linalg.solve(M, C)
    return [omega1, omega2, accelerations[0], accelerations[1]]

# --------------------------
# 2b: Hamiltonian
# --------------------------
def hamiltonian(state, L1, L2, m1, m2, g):
    theta1, theta2, p1, p2 = state
    # Generalized momenta from Lagrangian
    delta = theta2 - theta1
    H = (p1**2 * m2 * L2**2 + p2**2 * (m1 + m2) * L1**2 
         - 2 * p1 * p2 * m2 * L1 * L2 * np.cos(delta)) / (2 * L1**2 * L2**2 * m2 * (m1 + m2 * np.sin(delta)**2))
    H += -(m1 + m2)*g*L1*np.cos(theta1) - m2*g*L2*np.cos(theta2)
    return H

# --------------------------
# 2c: Phase Space Dynamics
# --------------------------
def double_pendulum_ode(t, state, L1, L2, m1, m2, g):
    theta1, theta2, p1, p2 = state
    # Hamiltonian equations (simplified)
    dtheta1 = p1 / (m1 * L1**2)
    dtheta2 = p2 / (m2 * L2**2)
    dp1 = ...  # Derived from Hamiltonian
    dp2 = ...  # Derived from Hamiltonian
    return [dtheta1, dtheta2, dp1, dp2]

def simulate_pendulum(initial_state, t_span, params):
    sol = solve_ivp(
        double_pendulum_ode,
        t_span,
        initial_state,
        args=tuple(params),
        method='RK45',
        dense_output=True
    )
    return sol

# --------------------------
# 2d: Phase Space Density
# --------------------------
def plot_convex_hull(points):
    hull = ConvexHull(points)
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# ==================
# Main Execution
# ==================
if __name__ == "__main__":
    # Parameters
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    params = (L1, L2, m1, m2, g)
    
    # Simulation
    initial_state = [np.pi/2, np.pi/2, 0.0, 0.0]  # Initial angles and momenta
    t_span = [0, 10]
    sol = simulate_pendulum(initial_state, t_span, params)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sol.y[0], sol.y[1], label=r'$\theta_1$ vs $\theta_2$')
    plt.xlabel(r'$\theta_2$')
    plt.ylabel(r'$p_2$')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.show()

