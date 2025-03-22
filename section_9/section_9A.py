import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad
from scipy.optimize import minimize
import matplotlib.animation as animation

# ==================================================
# Task 1: Black Body Radiation
# ==================================================

# Constants
k_B = 1.38064852e-23  # J/K
h = 6.626e-34          # J·s
c = 3e8                # m/s
ħ = h / (2 * np.pi)
prefactor = (k_B**4) / (c**2 * ħ**3 * 4 * np.pi**2)

# Part A: Integral with variable substitution
def integrand(z):
    x = z / (1 - z)
    jacobian = 1 / (1 - z)**2
    numerator = x**3
    denominator = np.exp(x) - 1
    return (numerator / denominator) * jacobian

integral_A, _ = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)
sigma_A = prefactor * 2 * np.pi * integral_A

# Part B: Using fixed_quad
def integrand_x(x):
    return x**3 / (np.exp(x) - 1)

integral_B = fixed_quad(integrand_x, 0, np.inf, n=50)[0]
sigma_B = prefactor * 2 * np.pi * integral_B

# Part C: Using quad with infinite limits
integral_C, _ = quad(integrand_x, 0, np.inf)
sigma_C = prefactor * 2 * np.pi * integral_C

print(f"Stefan-Boltzmann Constant (A): {sigma_A:.3e} W/m²K⁴")
print(f"Stefan-Boltzmann Constant (B): {sigma_B:.3e} W/m²K⁴")
print(f"Stefan-Boltzmann Constant (C): {sigma_C:.3e} W/m²K⁴")

# ==================================================
# Task 2: Planetary Orbits
# ==================================================

def hamiltonian_derivatives(q, p):
    r = np.sqrt(q[0]**2 + q[1]**2)
    dqdt = p
    dpdt = -q / r**3
    return dqdt, dpdt

# Initial conditions
e = 0.6
q0 = np.array([1 - e, 0.0])
p0 = np.array([0.0, np.sqrt((1 + e)/(1 - e))])

# Part A: Explicit Euler
def explicit_euler(T_f, n_steps):
    dt = T_f / n_steps
    q = np.zeros((n_steps+1, 2))
    p = np.zeros((n_steps+1, 2))
    q[0] = q0
    p[0] = p0
    
    for i in range(n_steps):
        dq, dp = hamiltonian_derivatives(q[i], p[i])
        q[i+1] = q[i] + dt * dq
        p[i+1] = p[i] + dt * dp
        
    return q

# Part B: Symplectic Euler
def symplectic_euler(T_f, n_steps):
    dt = T_f / n_steps
    q = np.zeros((n_steps+1, 2))
    p = np.zeros((n_steps+1, 2))
    q[0] = q0
    p[0] = p0
    
    for i in range(n_steps):
        # Update momentum first
        dp = -dt * q[i] / (np.linalg.norm(q[i])**3)
        p_new = p[i] + dp
        # Update position
        q_new = q[i] + dt * p_new
        q[i+1] = q_new
        p[i+1] = p_new
        
    return q

# Simulate and plot
q_explicit = explicit_euler(200, 100000)
q_symplectic = symplectic_euler(200, 400000)

plt.figure(figsize=(10, 6))
plt.plot(q_explicit[:, 0], q_explicit[:, 1], label='Explicit Euler')
plt.plot(q_symplectic[:, 0], q_symplectic[:, 1], label='Symplectic Euler')
plt.xlabel('q₁'); plt.ylabel('q₂'); plt.title('Planetary Orbits')
plt.legend(); plt.grid(); plt.show()

# ==================================================
# Task 3: Optimization Methods
# ==================================================

def H(theta):
    return theta**4 - 8*theta**2 - 2*np.cos(4*np.pi*theta)

def grad_H(theta):
    return 4*theta**3 - 16*theta + 8*np.pi*np.sin(4*np.pi*theta)

# Part A: Gradient Descent
def gradient_descent(theta0, alpha=0.01, max_iter=1000):
    theta = theta0
    history = [theta]
    for _ in range(max_iter):
        theta = theta - alpha * grad_H(theta)
        history.append(theta)
    return np.array(history)

# Part B: Metropolis-Hastings
def metropolis_hastings(theta0, beta=1.0, sigma=0.1, n_steps=10000):
    theta = theta0
    history = [theta]
    for _ in range(n_steps):
        theta_star = theta + np.random.normal(0, sigma)
        delta_H = H(theta_star) - H(theta)
        if delta_H < 0 or np.random.rand() < np.exp(-beta * delta_H):
            theta = theta_star
        history.append(theta)
    return np.array(history)

# Part C: Simulated Annealing
def simulated_annealing(theta0, beta0=1.0, delta_beta=0.01, n_steps=10000):
    beta = beta0
    theta = theta0
    history = [theta]
    for _ in range(n_steps):
        theta_star = theta + np.random.normal(0, 0.1)
        delta_H = H(theta_star) - H(theta)
        if delta_H < 0 or np.random.rand() < np.exp(-beta * delta_H):
            theta = theta_star
        beta += delta_beta
        history.append(theta)
    return np.array(history)

# Run optimizations
initial_guesses = [-1.0, 0.5, 3.0]
theta_range = np.linspace(-3, 3, 400)

plt.figure(figsize=(15, 5))
plt.plot(theta_range, H(theta_range), label='H(θ)')

# Plot optimization paths
colors = ['r', 'g', 'b']
for i, theta0 in enumerate(initial_guesses):
    # Gradient Descent
    gd_path = gradient_descent(theta0, alpha=0.001)
    plt.scatter(gd_path, H(gd_path), c=colors[i], s=10, label=f'GD θ0={theta0}')
    
    # Metropolis-Hastings
    mh_path = metropolis_hastings(theta0, beta=2.0)
    plt.scatter(mh_path, H(mh_path), c=colors[i], s=10, alpha=0.3, label=f'MH θ0={theta0}')
    
    # Simulated Annealing
    sa_path = simulated_annealing(theta0, beta0=0.1, delta_beta=0.001)
    plt.scatter(sa_path, H(sa_path), c=colors[i], s=10, marker='x', label=f'SA θ0={theta0}')

plt.xlabel('θ'); plt.ylabel('H(θ)'); plt.title('Optimization Paths')
plt.legend(); plt.grid(); plt.show()
