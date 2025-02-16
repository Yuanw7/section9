import numpy as np
import matplotlib.pyplot as plt
from TuringMachine import TuringMachine  

def random_binary(length):
    return ''.join(np.random.choice(['0', '1'], size=length))

def analyze_complexity(A_len, B_len, trials=10):
    n_list = []
    for _ in range(trials):
        A = random_binary(A_len)
        B = random_binary(B_len)
        tm = TuringMachine(A, B)
        tm.transition()  # Run until halting
        n_list.append(tm.step_count)  # store counts

    # Save histogram
    plt.figure()
    plt.hist(n_list, bins=20, edgecolor='black')
    plt.xlabel('Steps (n)')
    plt.ylabel('Frequency')
    plt.title(f'La,b = [{A_len}, {B_len}]')
    plt.savefig(f'hist_{A_len}_{B_len}.png')
    plt.close()
    
    # Return statistics
    return max(n_list), min(n_list), np.mean(n_list)

# Test all specified La,b pairs
pairs = [(2, 3), (3, 2), (3, 5), (5, 3), (3, 12), (12, 3)]

# Write results to a file
with open('part_d_results.txt', 'w') as f:
    for a, b in pairs:
        max_n, min_n, avg_n = analyze_complexity(a, b)
        f.write(f"La,b = [{a}, {b}]:\n")
        f.write(f"  - Worst (max): {max_n}\n")
        f.write(f"  - Best (min): {min_n}\n")
        f.write(f"  - Average: {avg_n:.2f}\n\n")
