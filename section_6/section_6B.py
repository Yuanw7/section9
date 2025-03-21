import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1  # Drift coefficient
sigma = 0.2  # Diffusion coefficient
t_final = 10  # Final time
N = 100  # Number of time steps
dt = t_final / N  # Time step size
np.random.seed(42)  # Random seed for reproducibility

# Wiener process increments
dW = np.sqrt(dt) * np.random.normal(size=N)

# Geometric Brownian Motion (Ito)
def simulate_gbm_ito(mu, sigma, t_final, N):
    """
    Simulate Geometric Brownian Motion using the Ito formulation.
    """
    t = np.linspace(0, t_final, N)
    X = np.zeros(N)
    X[0] = 1.0  # Initial condition
    for i in range(1, N):
        X[i] = X[i-1] * (1 + (mu + 0.5 * sigma**2) * dt + sigma * dW[i-1])
    return t, X

# Geometric Brownian Motion (Stratonovich)
def simulate_gbm_stratonovich(mu, sigma, t_final, N):
    """
    Simulate Geometric Brownian Motion using the Stratonovich formulation.
    """
    t = np.linspace(0, t_final, N)
    X = np.zeros(N)
    X[0] = 1.0  # Initial condition
    for i in range(1, N):
        X[i] = X[i-1] * (1 + mu * dt + sigma * dW[i-1])
    return t, X

# Simulate Ito and Stratonovich
t, X_ito = simulate_gbm_ito(mu, sigma, t_final, N)
_, X_strat = simulate_gbm_stratonovich(mu, sigma, t_final, N)

# Plot trajectories
plt.figure(figsize=(12, 6))
plt.plot(t, X_ito, label="Ito")
plt.plot(t, X_strat, label="Stratonovich")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Geometric Brownian Motion: Ito vs Stratonovich")
plt.legend()
plt.show()

# Statistics as a function of N
N_values = np.logspace(1, 4, num=20, dtype=int)  # Log-spaced values of N
means_ito, vars_ito = [], []
means_strat, vars_strat = [], []

for n in N_values:
    dt = t_final / n
    dW = np.sqrt(dt) * np.random.normal(size=n)
    _, X_ito = simulate_gbm_ito(mu, sigma, t_final, n)
    _, X_strat = simulate_gbm_stratonovich(mu, sigma, t_final, n)
    means_ito.append(np.mean(X_ito))
    vars_ito.append(np.var(X_ito))
    means_strat.append(np.mean(X_strat))
    vars_strat.append(np.var(X_strat))

# Plot statistics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.semilogx(N_values, means_ito, label="Ito Mean")
plt.xlabel("N")
plt.ylabel("Mean")
plt.legend()

plt.subplot(2, 2, 2)
plt.semilogx(N_values, vars_ito, label="Ito Variance")
plt.xlabel("N")
plt.ylabel("Variance")
plt.legend()

plt.subplot(2, 2, 3)
plt.semilogx(N_values, means_strat, label="Stratonovich Mean")
plt.xlabel("N")
plt.ylabel("Mean")
plt.legend()

plt.subplot(2, 2, 4)
plt.semilogx(N_values, vars_strat, label="Stratonovich Variance")
plt.xlabel("N")
plt.ylabel("Variance")
plt.legend()

plt.tight_layout()
plt.show()

# Functional Dynamics on GBM: f(X_t) = X_t^2
def simulate_functional_ito(mu, sigma, t_final, N):
    """
    Simulate the functional dynamics f(X_t) = X_t^2 using the Ito formulation.
    """
    t = np.linspace(0, t_final, N)
    X = np.zeros(N)
    F = np.zeros(N)
    X[0] = 1.0  # Initial condition
    for i in range(1, N):
        X[i] = X[i-1] * (1 + (mu + 0.5 * sigma**2) * dt + sigma * dW[i-1])
        F[i] = F[i-1] + X[i-1]**2 * (X[i] - X[i-1])
    return t, F

def simulate_functional_stratonovich(mu, sigma, t_final, N):
    """
    Simulate the functional dynamics f(X_t) = X_t^2 using the Stratonovich formulation.
    """
    t = np.linspace(0, t_final, N)
    X = np.zeros(N)
    F = np.zeros(N)
    X[0] = 1.0  # Initial condition
    for i in range(1, N):
        X[i] = X[i-1] * (1 + mu * dt + sigma * dW[i-1])
        F[i] = F[i-1] + X[i-1]**2 * (X[i] - X[i-1])
    return t, F

# Simulate functional dynamics
t, F_ito = simulate_functional_ito(mu, sigma, t_final, N)
_, F_strat = simulate_functional_stratonovich(mu, sigma, t_final, N)

# Plot functional dynamics
plt.figure(figsize=(12, 6))
plt.plot(t, F_ito, label="Ito Functional")
plt.plot(t, F_strat, label="Stratonovich Functional")
plt.xlabel("Time")
plt.ylabel("F(t)")
plt.title("Functional Dynamics: f(X_t) = X_t^2")
plt.legend()
plt.show()

# Autocorrelation function
def autocorrelation(F, t_lag):
    """
    Compute the autocorrelation function for a given time lag.
    """
    return np.correlate(F, F, mode='full')[len(F)-1-t_lag:len(F)-1+t_lag]

# Compute autocorrelation for F(t) at t = 5, 10, 20, 30
t_lags = [5, 10, 20, 30]
plt.figure(figsize=(12, 8))
for t_lag in t_lags:
    corr = autocorrelation(F_ito, t_lag)
    plt.plot(np.arange(-t_lag, t_lag), corr, label=f"t = {t_lag}")
plt.xlabel("Time Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function for F(t)")
plt.legend()
plt.show()
