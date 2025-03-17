import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.misc import derivative

# =============================================
# Task 1: Fermi-Dirac Statistics (Grand Canonical)
# =============================================

def fermi_grand_partition(mu, beta, epsilon):
    """Grand partition function for 2-level fermionic system"""
    return (1 + np.exp(beta * mu)) * (1 + np.exp(beta * (mu - epsilon)))

# =============================================
# Task 2: Bose-Einstein Condensate Analysis
# =============================================

# --------------------------
# 2a-b: Classical Partition Function
# --------------------------
def classical_partition(N, beta, epsilon):
    """Canonical partition function with binomial coefficients"""
    return sum([np.math.comb(N, n) * np.exp(-beta * n * epsilon) for n in range(N+1)])

def classical_average_occupation(N, beta, epsilon):
    """2c: Classical average occupation numbers"""
    Z = classical_partition(N, beta, epsilon)
    n1_avg = sum([n * np.math.comb(N, n) * np.exp(-beta * n * epsilon) for n in range(N+1)]) / Z
    return N - n1_avg, n1_avg

# --------------------------
# 2d-e: Quantum Partition Function
# --------------------------
def quantum_partition(N, beta, epsilon):
    """Canonical quantum partition function"""
    return sum([np.exp(-beta * n * epsilon) for n in range(N+1)])

def quantum_average_occupation(N, beta, epsilon):
    """2e: Quantum average occupation numbers"""
    Z = quantum_partition(N, beta, epsilon)
    n1_avg = sum([n * np.exp(-beta * n * epsilon) for n in range(N+1)]) / Z
    return N - n1_avg, n1_avg

# --------------------------
# 2f-h: Grand Canonical Formalism
# --------------------------
def grand_partition(mu, beta, epsilon):
    """Grand canonical partition function for bosons"""
    return 1 / ((1 - np.exp(beta * mu)) * (1 - np.exp(beta * (mu - epsilon))))

def avg_occupation(mu, beta, epsilon):
    """Average occupation numbers in grand canonical"""
    n0 = 1 / (np.exp(-beta * mu) - 1)
    n1 = 1 / (np.exp(-beta * (mu - epsilon)) - 1)
    return n0, n1

def solve_mu(N, beta, epsilon, mu_guess=-0.1):
    """Solve for chemical potential given N"""
    def equation(mu):
        n0, n1 = avg_occupation(mu, beta, epsilon)
        return n0 + n1 - N
    return fsolve(equation, mu_guess)[0]

# --------------------------
# 2i-k: BEC Analysis
# --------------------------
class BoseSystem:
    def __init__(self, N=1e5, epsilon=1.0, deg=100):
        self.N = N
        self.epsilon = epsilon
        self.levels = np.linspace(0, epsilon, deg)  # Near-degenerate levels
        
    def total_occupation(self, mu, beta):
        """Total occupation number for given mu"""
        return sum([1/(np.exp(beta * (e - mu)) - 1) for e in self.levels])
    
    def solve_mu(self, T, mu_guess=-0.1):
        beta = 1/(k_B*T) if T != 0 else np.inf
        def equation(mu):
            return self.total_occupation(mu, beta) - self.N
        return fsolve(equation, mu_guess)[0]
    
    def thermodynamic_quantities(self, T):
        beta = 1/(k_B*T) if T != 0 else np.inf
        mu = self.solve_mu(T)
        n0 = 1/(np.exp(beta * (0 - mu)) - 1)
        log_n0 = np.log(n0) if n0 > 0 else -np.inf
        dn0dT = derivative(lambda t: 1/(np.exp(1/(k_B*t)*(mu)) - 1), T, dx=1e-6)
        Cv = ...  # Specific heat calculation
        return mu, n0, log_n0, dn0dT, Cv

# ==================
# Constants and Parameters
# ==================
k_B = 1.380649e-23  # Boltzmann constant
N = 1e5
epsilon = 1.0
temps = np.linspace(1e-6, 2.0, 100)

# ==================
# Main Execution
# ==================
if __name__ == "__main__":
    # Initialize BEC system
    bec_system = BoseSystem(N=N, epsilon=epsilon)
    
    # Calculate thermodynamic quantities
    results = [bec_system.thermodynamic_quantities(T) for T in temps]
    mu_vals, n0_vals, log_n0, dn0dT, Cv = zip(*results)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.plot(temps, mu_vals)
    plt.xlabel('Temperature')
    plt.ylabel('Chemical Potential (μ)')
    
    plt.subplot(222)
    plt.plot(temps, n0_vals)
    plt.yscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('log(n₀)')
    
    plt.subplot(223)
    plt.plot(temps, dn0dT)
    plt.xlabel('Temperature')
    plt.ylabel('∂n₀/∂T')
    
    plt.subplot(224)
    plt.plot(temps, Cv)
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat (Cv)')
    
    plt.tight_layout()
    plt.show()
