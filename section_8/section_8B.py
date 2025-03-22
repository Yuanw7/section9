import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import roots_legendre
from scipy.integrate import quad, fixed_quad, romberg

# ==================================================
# Task 1: Quadrature Methods
# ==================================================

class Quadrature:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def midpoint(self, f):
        """Midpoint Rule"""
        mid = (self.a + self.b) / 2
        return (self.b - self.a) * f(mid)

    def trapezoidal(self, f):
        """Trapezoidal Rule"""
        return (self.b - self.a) / 2 * (f(self.a) + f(self.b))

    def simpson(self, f):
        """Simpson's Rule"""
        mid = (self.a + self.b) / 2
        return (self.b - self.a) / 6 * (f(self.a) + 4 * f(mid) + f(self.b))


class GaussQuad(Quadrature):
    def __init__(self, a, b, order):
        super().__init__(a, b)
        self.order = order

    def legendre_poly(self, x, M):
        """Legendre Polynomial of order M"""
        if M == 0:
            return 1
        elif M == 1:
            return x
        else:
            return ((2 * M - 1) * x * self.legendre_poly(x, M - 1) - (M - 1) * self.legendre_poly(x, M - 2)) / M

    def gauss_legendre(self, f):
        """Gauss-Legendre Quadrature"""
        x, w = roots_legendre(self.order)
        x_scaled = (self.b - self.a) / 2 * x + (self.a + self.b) / 2
        return (self.b - self.a) / 2 * np.sum(w * f(x_scaled))


# ==================================================
# Task 2: Quadrature on Test Functions
# ==================================================

def polynomial_integral(a, b, k):
    """True integral of x^k from a to b"""
    return (b**(k+1) - a**(k+1)) / (k+1)

def fermi_dirac_integral(a, b, k):
    """True integral of Fermi-Dirac distribution from a to b"""
    return (np.log(np.exp(k * b) + 1) - np.log(np.exp(k * a) + 1)) / k

def relative_error(true, approx):
    """Relative error calculation"""
    return 2 * abs(true - approx) / (true + approx)

def generate_heatmap(quad_method, true_integral, f, a, b, k_values, N_values):
    """Generate heatmap of relative errors"""
    errors = np.zeros((len(k_values), len(N_values)))
    for i, k in enumerate(k_values):
        for j, N in enumerate(N_values):
            integral = quad_method(f, a, b, N)
            errors[i, j] = relative_error(true_integral(a, b, k), integral)
    sns.heatmap(errors, xticklabels=N_values, yticklabels=k_values, annot=True)
    plt.xlabel("N")
    plt.ylabel("k")
    plt.title(f"Relative Error Heatmap for {quad_method.__name__}")
    plt.show()


# ==================================================
# Task 3: Harmonic Oscillator
# ==================================================

def time_period(a, m=1):
    """Time period for V(x) = x^4"""
    def integrand(x):
        return 1 / np.sqrt(a**4 - x**4)
    return np.sqrt(8 * m) * quad(integrand, 0, a)[0]

def time_period_fixed_quad(a, N):
    """Time period using fixed_quad"""
    def integrand(x):
        return 1 / np.sqrt(a**4 - x**4)
    return np.sqrt(8) * fixed_quad(integrand, 0, a, n=N)[0]

def time_period_romberg(a, divmax=10):
    """Time period using romberg"""
    def integrand(x):
        return 1 / np.sqrt(a**4 - x**4)
    return np.sqrt(8) * romberg(integrand, 0, a, divmax=divmax, show=True)


# ==================================================
# Main Script
# ==================================================

if __name__ == "__main__":
    # Parameters
    a, b = 0, 1
    k = 2
    f_poly = lambda x: x**k
    f_fermi = lambda x: 1 / (1 + np.exp(-k * x))

    # Quadrature example
    quad = Quadrature(a, b)
    gauss = GaussQuad(a, b, order=5)

    print("Midpoint (Polynomial):", quad.midpoint(f_poly))
    print("Gauss-Legendre (Polynomial):", gauss.gauss_legendre(f_poly))

    # Plot Legendre Polynomials
    x = np.linspace(-1, 1, 1000)
    plt.figure(figsize=(10, 6))
    for M in range(1, 6):
        y = [gauss.legendre_poly(xi, M) for xi in x]
        plt.plot(x, y, label=f"M={M}")
    plt.xlabel("x")
    plt.ylabel("P_M(x)")
    plt.title("Legendre Polynomials")
    plt.legend()
    plt.grid()
    plt.show()

    # Heatmap of relative errors
    k_values = np.arange(0, 11)
    N_values = np.logspace(1, 5, num=10, dtype=int)
    generate_heatmap(quad.midpoint, polynomial_integral, f_poly, a, b, k_values, N_values)

    # Harmonic Oscillator
    a = 2
    print("Time Period (a=2):", time_period(a))

    # Error estimation with fixed_quad
    N = 10
    error = abs(time_period_fixed_quad(a, N) - time_period_fixed_quad(a, 2 * N))
    print("Error for N=10:", error)

    # Romberg integration
    print("Time Period (Romberg):", time_period_romberg(a))

    # Plot Time Period vs Amplitude
    a_values = np.linspace(0, 2, 100)
    T_values = [time_period(a) for a in a_values]

    plt.figure(figsize=(10, 6))
    plt.plot(a_values, T_values, label="Time Period")
    plt.xlabel("Amplitude (a)")
    plt.ylabel("Time Period (T)")
    plt.title("Time Period vs Amplitude")
    plt.grid()
    plt.legend()
    plt.show()
