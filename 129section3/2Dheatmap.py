import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from TuringMachine import TuringMachine

def random_binary(length):
    """Generate a random binary string of given length."""
    return ''.join(np.random.choice(['0', '1'], size=length))

def compute_heatmap(max_length=30, trials=5):
    """Generate average steps for a grid of La,b pairs."""
    heatmap = np.zeros((max_length-1, max_length-1))  # 2-30 → indices 0-28
    
    for a in range(2, max_length+1):
        for b in range(2, max_length+1):
            steps = []
            for _ in range(trials):
                A = random_binary(a)
                B = random_binary(b)
                tm = TuringMachine(A, B)
                tm.transition()
                steps.append(tm.step_count)
            avg_steps = np.mean(steps)
            heatmap[a-2][b-2] = avg_steps  # Adjust indices
    
    return heatmap

# Generate heatmap data
heatmap_data = compute_heatmap(max_length=30, trials=5)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    annot=False,
    cmap='viridis',
    xticklabels=np.arange(2, 31),
    yticklabels=np.arange(2, 31)
)
plt.xlabel("b (Length of B)")
plt.ylabel("a (Length of A)")
plt.title("Average Computation Complexity ⟨n⟩ for La,b = [a, b]")
plt.savefig("heatmap.png")
plt.close()
