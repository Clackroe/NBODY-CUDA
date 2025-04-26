import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv('data.csv')

# Separate data
speedup_row = df[df['Method'] == 'CUDA Speedup']
runtime_df = df[df['Method'] != 'CUDA Speedup']

sizes = df.columns[1:].astype(int).to_numpy()
methods = runtime_df['Method'].to_numpy()
times = runtime_df.iloc[:, 1:].to_numpy()
speedup = speedup_row.iloc[:, 1:].to_numpy().flatten()

# Set up plots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 1. Runtime comparison: CUDA (GPU) vs OMP (CPU)
bar_width = 0.3
x = np.arange(len(sizes))

for idx, method in enumerate(methods):
    axs[0].bar(x + idx * bar_width, times[idx], width=bar_width, label=method)

axs[0].set_xlabel('Num Bodies')
axs[0].set_ylabel('Time (seconds)')
axs[0].set_title('Runtime Comparison (CUDA vs OMP) (Lower is better)')
axs[0].set_xticks(x + bar_width / 2)
axs[0].set_xticklabels(sizes)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# 2. CUDA Speedup plot
axs[1].plot(sizes, speedup, marker='o', color='green', linewidth=2)
axs[1].set_xlabel('Num Bodies')
axs[1].set_ylabel('Speedup Factor (x)')
axs[1].set_title('CUDA Speedup over CPU (Higher is better)')
axs[1].set_xticks(sizes)
axs[1].set_xticklabels(sizes)
axs[1].grid(True, linestyle='--', alpha=0.7)

# Annotate each speedup point
for i, val in enumerate(speedup):
    axs[1].annotate(f"{val:.2f}x", (sizes[i], speedup[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Layout
plt.subplots_adjust(top=10)
plt.tight_layout()
if (os.path.exists("data.png")):
    os.remove("data.png")
plt.savefig("data.png")
plt.show()
