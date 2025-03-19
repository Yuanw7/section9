#question 1
def construct_site_transition_matrix():
    """
    Construct the 8x8 transition matrix for N=3 Heisenberg model in site basis.
    """
    P = np.zeros((8, 8))
    # State indices: 0=|↑↑↑⟩, 1=|↑↑↓⟩, 2=|↑↓↑⟩, 3=|↑↓↓⟩, 4=|↓↑↑⟩, 5=|↓↑↓⟩, 6=|↓↓↑⟩, 7=|↓↓↓⟩
    transitions = [
        [0],                   # State 0
        [2, 4],                # State 1
        [1, 4],                # State 2
        [5, 6],                # State 3
        [1, 2],                # State 4
        [3, 6],                # State 5
        [3, 5],                # State 6
        [7]                    # State 7
    ]
    for i in range(8):
        if len(transitions[i]) == 0:
            P[i, i] = 1.0
        else:
            prob = 1.0 / len(transitions[i])
            for j in transitions[i]:
                P[i, j] = prob
    return P

P_site = construct_site_transition_matrix()
print("Site Transition Matrix:\n", P_site)

# question 2
def find_stationary_distribution(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1.0)]
    stationary = stationary[:, 0].real
    stationary /= stationary.sum()
    return stationary

pi_site = find_stationary_distribution(P_site)
print("Stationary Distribution (Site):", pi_site)
#question 4

def construct_magnon_transition_matrix(J, T, N=3):
    k_values = np.arange(N)
    E = 2 * J * np.sin(np.pi * k_values / N)**2
    P = np.zeros((N, N))
    for i in range(N):
        energies = E - E[i]
        weights = np.exp(-energies / T)
        weights[i] = 0  # Exclude self-transitions (if needed)
        total = weights.sum()
        if total == 0:
            P[i, i] = 1.0
        else:
            P[i, :] = weights / total
    return P

J = 1.0
T = 1.0
P_magnon = construct_magnon_transition_matrix(J, T)
print("Magnon Transition Matrix:\n", P_magnon)

#question 5

pi_magnon = find_stationary_distribution(P_magnon)
print("Stationary Distribution (Magnon):", pi_magnon)

# question 6

# Initial guesses for magnon basis
initial_m1 = np.array([1, 0, 0])
initial_m2 = np.array([0.5, 0, 0.5])
initial_m3 = np.ones(3) / 3

pi_m1 = power_iteration(P_magnon, initial_m1)
pi_m2 = power_iteration(P_magnon, initial_m2)
pi_m3 = power_iteration(P_magnon, initial_m3)

print("Magnon Power Iteration Results:")
print("Initial 1:", pi_m1)
print("Initial 2:", pi_m2)
print("Initial 3:", pi_m3)

#question 7

from scipy.integrate import solve_ivp

def construct_rate_matrix(P, dt=0.1):
    Q = np.log(P) / dt  # Approximation for small dt
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q

Q = construct_rate_matrix(P_magnon)

def master_equation(t, pi, Q):
    return pi @ Q

initial_condition = np.array([1, 0, 0])
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)
sol = solve_ivp(master_equation, t_span, initial_condition, args=(Q,), t_eval=t_eval, method='RK45')

print("Master Equation Solution:", sol.y[:, -1])

# Visualization
import matplotlib.pyplot as plt
plt.plot(sol.t, sol.y.T)
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Master Equation Evolution')
plt.legend(['|k=0⟩', '|k=1⟩', '|k=2⟩'])
plt.show()

