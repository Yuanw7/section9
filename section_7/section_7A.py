#part a

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta, gamma
from scipy.stats import beta as beta_dist

# Load datasets
def load_dataset(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data)

dataset_1 = load_dataset('dataset_1.json')
dataset_2 = load_dataset('dataset_2.json')
dataset_3 = load_dataset('dataset_3.json')

# Bayesian posterior calculation
def bayesian_posterior(data, batch_size=50):
    M = np.cumsum(data)
    N = np.arange(1, len(data)+1)
    posteriors = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        M_batch = np.sum(batch)
        N_batch = len(batch)
        # Beta posterior parameters: alpha = M+1, beta = N-M+1
        a = M_batch + 1
        b = N_batch - M_batch + 1
        posteriors.append((a, b))
    return posteriors

# Plot posteriors
def plot_posteriors(posteriors, title):
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    for a, b in posteriors:
        plt.plot(x, beta_dist.pdf(x, a, b), label=f'a={a}, b={b}')
    plt.xlabel('p')
    plt.ylabel('P(p|M,N)')
    plt.title(title)
    plt.legend()
    plt.show()

# Calculate expectation and variance
def posterior_stats(posteriors):
    stats = []
    for a, b in posteriors:
        mean = a / (a + b)
        var = (a * b) / ((a + b)**2 * (a + b + 1))
        stats.append((mean, var))
    return stats

# Example usage
posteriors_1 = bayesian_posterior(dataset_1)
plot_posteriors(posteriors_1, 'Dataset 1 Posterior')
stats_1 = posterior_stats(posteriors_1)
print("Dataset 1 - Mean:", [s[0] for s in stats_1], "\nVariance:", [s[1] for s in stats_1])

#part b
def stirling_approx(n):
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

N_values = np.arange(1, 11)
log_factorial = [np.log(gamma(n + 1)) for n in N_values]
stirling_values = [stirling_approx(n) for n in N_values]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(N_values, log_factorial, 'ro', label='Exact (Gamma)')
plt.plot(N_values, stirling_values, 'b-', label='Stirling')
plt.xlabel('N')
plt.ylabel('log(n!)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(N_values, np.array(log_factorial) - np.array(stirling_values), 'g-')
plt.xlabel('N')
plt.ylabel('Difference')
plt.title('Stirling Approximation Error')
plt.tight_layout()
plt.show()

#part e
def bootstrap(data, sample_sizes, n_bootstrap=100):
    results = {}
    for size in sample_sizes:
        means = []
        variances = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=size, replace=True)
            means.append(np.mean(sample))
            variances.append(np.var(sample))
        results[size] = (means, variances)
    return results

sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
bootstrap_results = bootstrap(dataset_1, sample_sizes)

# Plot histograms
plt.figure(figsize=(15, 10))
for i, size in enumerate(sample_sizes):
    plt.subplot(3, 3, i+1)
    plt.hist(bootstrap_results[size][0], bins=20)
    plt.title(f'Sample Size: {size}')
plt.tight_layout()
plt.show()

#task2 part a

from scipy.optimize import minimize

# Load decay data
vacuum_data = load_dataset('Vacuum_decay_dataset.json')
cavity_data = load_dataset('Cavity_decay_dataset.json')

# Truncated exponential PDF (x >= 1)
def truncated_exp_pdf(x, lambd):
    Z = np.exp(-1 / lambd)
    return (1 / (lambd * Z)) * np.exp(-x / lambd)

# Gaussian PDF
def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# Mixture model likelihood
def mixture_log_likelihood(params, x):
    lambd, mu, sigma, weight = params
    exp_part = weight * truncated_exp_pdf(x, lambd)
    gauss_part = (1 - weight) * gaussian_pdf(x, mu, sigma)
    return -np.sum(np.log(exp_part + gauss_part))

# Initial guess and bounds
initial_guess = [1.0, 5.0, 1.0, 0.5]
bounds = [(0.1, 10), (0, 20), (0.1, 5), (0, 1)]

# MLE optimization
result = minimize(mixture_log_likelihood, initial_guess, args=(cavity_data), bounds=bounds)
lambd_opt, mu_opt, sigma_opt, weight_opt = result.x

print(f"Optimal parameters: λ={lambd_opt}, μ={mu_opt}, σ={sigma_opt}, weight={weight_opt}")

#part b

from scipy.stats import chi2

# Null hypothesis (only exponential)
def exp_log_likelihood(lambd, x):
    return -np.sum(np.log(truncated_exp_pdf(x, lambd)))

result_null = minimize(exp_log_likelihood, 1.0, args=(cavity_data), bounds=[(0.1, 10)])
lambda_null = result_null.x[0]

# Likelihood ratio test
LR = 2 * (result_null.fun - result.fun)
p_value = 1 - chi2.cdf(LR, df=3)  # 3 additional parameters

print(f"LR: {LR}, p-value: {p_value}")
if p_value < 0.05:
    print("Reject null hypothesis (additional decay component exists)")
else:
    print("Fail to reject null hypothesis")
