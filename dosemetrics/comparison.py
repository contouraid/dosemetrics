import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# Define dose bins (0 to 100 Gy, 101 bins for 1 Gy resolution)
dose_bins = np.linspace(0, 100, 101)
bin_width = dose_bins[1] - dose_bins[0]  # 1 Gy

# --- Example 1: Different Mean Doses, Similar Spread ---
# Plan A: Mean 50 Gy, std 10 Gy
dvh_a1 = np.exp(-((dose_bins - 50)**2) / (2 * 10**2)) / (10 * np.sqrt(2 * np.pi))
dvh_a1 /= dvh_a1.sum()  # Normalize

# Plan B: Mean 55 Gy, std 10 Gy
dvh_b1 = np.exp(-((dose_bins - 55)**2) / (2 * 10**2)) / (10 * np.sqrt(2 * np.pi))
dvh_b1 /= dvh_b1.sum()  # Normalize

# Compute metrics for Example 1
area1 = np.sum(np.abs(dvh_a1 - dvh_b1)) * bin_width
w_distance1 = wasserstein_distance(dose_bins, dose_bins, dvh_a1, dvh_b1)

# --- Example 2: Similar Mean Doses, Different Maximum Doses (via spread) ---
# Plan A: Mean 50 Gy, std 8 Gy (narrower, lower max dose)
dvh_a2 = np.exp(-((dose_bins - 50)**2) / (2 * 8**2)) / (8 * np.sqrt(2 * np.pi))
dvh_a2 /= dvh_a2.sum()  # Normalize

# Plan B: Mean 50 Gy, std 12 Gy (broader, higher max dose)
dvh_b2 = np.exp(-((dose_bins - 50)**2) / (2 * 12**2)) / (12 * np.sqrt(2 * np.pi))
dvh_b2 /= dvh_b2.sum()  # Normalize

# Compute metrics for Example 2
area2 = np.sum(np.abs(dvh_a2 - dvh_b2)) * bin_width
w_distance2 = wasserstein_distance(dose_bins, dose_bins, dvh_a2, dvh_b2)

# --- Visualization ---
# Example 1 Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(dose_bins, dvh_a1, label='Plan A (Mean 50 Gy)', color='#1f77b4', linewidth=2)
plt.plot(dose_bins, dvh_b1, label='Plan B (Mean 55 Gy)', color='#ff7f0e', linewidth=2)
plt.fill_between(dose_bins, dvh_a1, dvh_b1, color='gray', alpha=0.3, label='Area Between')
plt.xlabel('Dose (Gy)', fontsize=12)
plt.ylabel('Volume Fraction', fontsize=12)
plt.title('Example 1: Different Mean Doses', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(10, max(dvh_a1)*0.9, f'Area: {area1:.4f}\nWasserstein: {w_distance1:.4f} Gy', fontsize=10)

# Example 2 Plot
plt.subplot(1, 2, 2)
plt.plot(dose_bins, dvh_a2, label='Plan A (Std 8 Gy)', color='#1f77b4', linewidth=2)
plt.plot(dose_bins, dvh_b2, label='Plan B (Std 12 Gy)', color='#ff7f0e', linewidth=2)
plt.fill_between(dose_bins, dvh_a2, dvh_b2, color='gray', alpha=0.3, label='Area Between')
plt.xlabel('Dose (Gy)', fontsize=12)
plt.ylabel('Volume Fraction', fontsize=12)
plt.title('Example 2: Different Maximum Doses', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(10, max(dvh_a2)*0.9, f'Area: {area2:.4f}\nWasserstein: {w_distance2:.4f} Gy', fontsize=10)

plt.tight_layout()
plt.savefig('dvh_comparison.png')