#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.patches import Rectangle, ConnectionPatch

# Set up the figure with the whitegrid style
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Find all precision files for finetuned models
precision_files = glob.glob("temp_data_files/precision_Finetuned-12_*.npy")

# Use a list to collect (ratio_value, label, recall_array, precision_array)
curves = []
for pf in precision_files:
    # Corresponding recall filename
    rf = pf.replace("precision_", "recall_")
    if not os.path.exists(rf):
        continue

    # Extract ratio string and value from filename
    m = re.search(r"precision_[^_]+_(\d+:\d+)\.npy$", os.path.basename(pf))
    if not m:
        continue
    label = m.group(1)          # e.g. "1:10"
    a, b = map(float, label.split(":"))
    ratio_value = a / b         # used for sorting

    # Load data
    precision = np.load(pf)
    recall = np.load(rf)
    
    # Apply LOWESS smoothing
    # Sort points by recall for smoothing
    sorted_pairs = sorted(zip(recall, precision))
    recall_sorted = np.array([p[0] for p in sorted_pairs])
    precision_sorted = np.array([p[1] for p in sorted_pairs])
    
    # Apply LOWESS smoothing - frac controls the smoothing window
    smoothed = lowess(precision_sorted, recall_sorted, frac=0.2, it=1, return_sorted=True)
    recall_smooth = smoothed[:, 0]
    precision_smooth = smoothed[:, 1]
    
    curves.append((ratio_value, label, recall_smooth, precision_smooth))

# Sort by ratio_value in descending order (1:2 highest, 1:10000 lowest)
curves.sort(key=lambda x: x[0], reverse=True)

# Use viridis color scheme
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(curves)))

# Create main figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("#f8f8f8")
fig.patch.set_facecolor("#f8f8f8")

# Plot the curves on the main axes
for i, (ratio_value, label, recall, precision) in enumerate(curves):
    ax.plot(recall, precision, linewidth=2.5, label=label.replace(":", "/"), 
            color=colors[i], alpha=0.9,
            solid_capstyle='round', solid_joinstyle='round')

# Axes and title for main plot
ax.set_xlabel("Recall", fontsize=28)
ax.set_ylabel("Precision", fontsize=28)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=22)

# Set custom ticks to show zero only on x-axis for the main plot
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])  # Skip zero on y-axis

ax.grid(True, linestyle="--", alpha=0.7)

# Legend in the upper right corner
ax.legend(fontsize=20, loc="upper right", frameon=True, framealpha=0.9, 
          edgecolor="black", title="Ratio", title_fontsize=20)

# Create the zoomed inset axes
# Position it in the middle-left area, moved up a bit
axins = ax.inset_axes([0.3, 0.65, 0.35, 0.35], facecolor="#f8f8f8")

# Plot the same curves on the inset axes
for i, (ratio_value, label, recall, precision) in enumerate(curves):
    axins.plot(recall, precision, linewidth=2.5, 
               color=colors[i], alpha=0.9,
               solid_capstyle='round', solid_joinstyle='round')

# Set the limits for the inset axes (zoomed region)
zoom_x_min, zoom_x_max = 0, 0.2
zoom_y_min, zoom_y_max = 0, 0.2
axins.set_xlim(zoom_x_min, zoom_x_max)
axins.set_ylim(zoom_y_min, zoom_y_max)
axins.tick_params(labelsize=16)

# Set custom ticks to avoid showing zero on both axes
# Show zero only on x-axis, not on y-axis
axins.set_xticks([0, 0.1, 0.2])
axins.set_yticks([0.1, 0.2])  # Skip zero on y-axis

axins.grid(True, linestyle="--", alpha=0.7)

# Remove spines from inset to match main plot style
for spine in axins.spines.values():
    spine.set_visible(False)

# Draw a rectangle on the main plot showing the zoom region
rect = Rectangle((zoom_x_min, zoom_y_min), zoom_x_max-zoom_x_min, zoom_y_max-zoom_y_min,
                 fill=False, edgecolor='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax.add_patch(rect)

# # Add connecting lines between the rectangle and the inset plot
# # Connect bottom-left corners
con1 = ConnectionPatch(xyA=(zoom_x_min, zoom_y_min), coordsA=ax.transData,
                      xyB=(zoom_x_min, zoom_y_min), coordsB=axins.transData,
                      linestyle="--", color="gray", alpha=0.3, linewidth=1.5)
fig.add_artist(con1)

# Connect top-left corners instead of top-right to avoid crossing the plot area
con2 = ConnectionPatch(xyA=(zoom_x_min, zoom_y_max), coordsA=ax.transData,
                      xyB=(zoom_x_min, zoom_y_max), coordsB=axins.transData,
                      linestyle="--", color="gray", alpha=0.3, linewidth=1.5)
fig.add_artist(con2)

plt.tight_layout()

# Save the combined plot
out_png = "finetuned_pr_curves_fractions_with_zoom.png"
plt.savefig(out_png, dpi=400, bbox_inches='tight', facecolor="#f8f8f8")
print(f"Saved PR plot with zoom inset to {out_png}")

out_pdf = "image_files/finetuned_pr_curves_fractions_with_zoom.pdf"
plt.savefig(out_pdf, bbox_inches='tight', facecolor="#f8f8f8")
print(f"Saved PR plot with zoom inset to {out_pdf}")
