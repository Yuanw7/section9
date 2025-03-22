# TASK 1

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 4
J = 1.0
B = 0.0
T = 1.0
beta = 1.0 / T

# Generate all possible spin configurations (L=4)
n_spins = L**2
configurations = []
energies = []

for i in range(2**n_spins):
    # Convert integer to binary spin configuration
    binary = bin(i)[2:].zfill(n_spins)
    spin = np.array([1 if c == '1' else -1 for c in binary], dtype=int)
    spin_grid = spin.reshape(L, L)
    
    # Compute energy (periodic boundary conditions)
    right = spin_grid * np.roll(spin_grid, shift=1, axis=1)
    down = spin_grid * np.roll(spin_grid, shift=1, axis=0)
    energy = -J * (np.sum(right) + np.sum(down)) - B * np.sum(spin_grid)
    energies.append(energy)
    configurations.append(spin_grid)

# Partition function and probabilities
Z = np.sum(np.exp(-beta * np.array(energies)))
probabilities = np.exp(-beta * np.array(energies)) / Z

# Sample 5 configurations
cum_probs = np.cumsum(probabilities)
samples_idx = [np.searchsorted(cum_probs, np.random.rand()) for _ in range(5)]
samples = [configurations[i] for i in samples_idx]

# Plot sampled configurations
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(samples[i], cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
plt.suptitle("Sampled Spin Configurations (L=4)")
plt.show()

# Plot PDF of energies
unique_energies, counts = np.unique(energies, return_counts=True)
energy_probs = [np.sum(probabilities[np.array(energies) == e) for e in unique_energies]

plt.figure(figsize=(10, 6))
plt.bar(unique_energies, energy_probs)
plt.xlabel("Energy")
plt.ylabel("Probability")
plt.title("Energy Distribution (L=4, T=1)")
plt.show()

#TASK 2
def gibbs_sampler(L, T, n_steps=1000, burn_in=200):
    beta = 1.0 / T
    spins = np.random.choice([-1, 1], size=(L, L))
    magnetization = []
    
    for step in range(n_steps + burn_in):
        for i in range(L):
            for j in range(L):
                # Neighbors with periodic boundary conditions
                top = spins[(i-1)%L, j]
                bottom = spins[(i+1)%L, j]
                left = spins[i, (j-1)%L]
                right = spins[i, (j+1)%L]
                sum_neighbors = top + bottom + left + right
                p = 1 / (1 + np.exp(-2 * beta * J * sum_neighbors))
                spins[i, j] = 1 if np.random.rand() < p else -1
        
        if step >= burn_in:
            magnetization.append(np.mean(spins))
    
    return np.mean(magnetization)

# Critical temperature
Tc = 2 * J / np.log(1 + np.sqrt(2))

# Simulate magnetization for different T and L
Ls = [10, 17, 25, 32, 40]
T_values = np.linspace(1.0, 4.0, 20)
results = {L: [] for L in Ls}

for L in Ls:
    for T in T_values:
        M = gibbs_sampler(L, T, n_steps=1000, burn_in=200)
        results[L].append(M)

# Plot Magnetization vs Temperature
plt.figure(figsize=(10, 6))
for L in Ls:
    plt.plot(T_values, results[L], marker='o', label=f"L={L}")

plt.axvline(Tc, color='red', linestyle='--', label=f"T_c={Tc:.2f}")
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Phase Transition in 2D Ising Model")
plt.legend()
plt.show()
