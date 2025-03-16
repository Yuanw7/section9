# part a
def find_fixed_points(r):
    fixed_points = []
    # x = 0 is always a fixed point
    fixed_points.append(0)
    # x = 1 - 1/r if r != 0
    if r != 0:
        x = 1 - 1/r
        if 0 <= x <= 1:
            fixed_points.append(x)
    return fixed_points

def stability(r, x):
    derivative = r * (1 - 2 * x)
    if abs(derivative) < 1:
        return "Stable"
    elif abs(derivative) == 1:
        return "Neutral"
    else:
        return "Unstable"

# Test for r = 1, 2, 3, 4
for r in [1, 2, 3, 4]:
    fps = find_fixed_points(r)
    print(f"r = {r}:")
    for x in fps:
        print(f"  Fixed point at x = {x:.2f}: {stability(r, x)}")

# part b
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, threshold=1e-6, max_iter=1000):
    x = [x0]
    for _ in range(max_iter):
        xn = r * x[-1] * (1 - x[-1])
        if abs(xn - x[-1]) < threshold:
            break
        x.append(xn)
    return x

# Test for r = 2, 3, 3.5, 3.8, 4.0
r_values = [2, 3, 3.5, 3.8, 4.0]
x0 = 0.2
results = {}

for r in r_values:
    trajectory = logistic_map(r, x0)
    results[r] = trajectory
    print(f"r = {r}: Converged to {trajectory[-1]:.4f} in {len(trajectory)} iterations")

# Plotting
plt.figure(figsize=(10, 6))
for r in r_values:
    plt.plot(results[r], label=f'r = {r}')
plt.xlabel('Iteration')
plt.ylabel('x_n')
plt.title('Logistic Map Time Series')
plt.legend()
plt.show()

# part c
x0_values = [0.1, 0.3, 0.5]
r_test = 3.8

plt.figure(figsize=(10, 6))
for x0 in x0_values:
    trajectory = logistic_map(r_test, x0)
    plt.plot(trajectory, label=f'x0 = {x0}')

plt.xlabel('Iteration')
plt.ylabel('x_n')
plt.title(f'Logistic Map for r = {r_test} with Different Initial Conditions')
plt.legend()
plt.show()

#part d

def bifurcation_diagram(r_min=0, r_max=4, dr=0.01, x0=0.2, n_transient=500, n_iter=100):
    r_values = np.arange(r_min, r_max + dr, dr)
    bifurcation = []

    for r in r_values:
        x = x0
        # Transient iterations
        for _ in range(n_transient):
            x = r * x * (1 - x)
        # Collect data after transient
        x_vals = []
        for _ in range(n_iter):
            x = r * x * (1 - x)
            x_vals.append(x)
        bifurcation.append((r, x_vals))

    return bifurcation

# Generate data
bifurcation_data = bifurcation_diagram()

# Plot
plt.figure(figsize=(12, 8))
for r, x_vals in bifurcation_data:
    plt.plot([r] * len(x_vals), x_vals, ',k', alpha=0.1)
plt.xlabel('r')
plt.ylabel('x_n (steady states)')
plt.title('Bifurcation Diagram of the Logistic Map')
plt.show()

#part e

def modified_logistic(r, gamma, x0=0.2, threshold=1e-6):
    x = x0
    for _ in range(1000):
        xn = r * x * (1 - x**gamma)
        if abs(xn - x) < threshold:
            return r
        x = xn
    return r

# Find first bifurcation point for each gamma
gammas = np.linspace(0.5, 1.5, 50)
critical_rs = []

for gamma in gammas:
    # Binary search for bifurcation
    r_low, r_high = 2.0, 4.0
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        if modified_logistic(r_mid, gamma) is None:
            r_high = r_mid
        else:
            r_low = r_mid
    critical_rs.append(r_mid)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(gammas, critical_rs, 'b-')
plt.xlabel('gamma')
plt.ylabel('First Bifurcation r')
plt.title('First Bifurcation Point vs. gamma')
plt.grid(True)
plt.show()

