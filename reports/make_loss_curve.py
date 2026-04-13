#!/usr/bin/env python3
"""Generate Hokusai training loss curve chart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Verified data from train_20260412_180037_gpu0.log
steps = [50, 100, 200, 400, 600, 800, 1000, 1200]
losses = [0.142, 0.129, 0.128, 0.121, 0.126, 0.133, 0.127, 0.123]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, losses, "o-", color="#2c5f8a", linewidth=2, markersize=6)

# Annotate minimum
ax.annotate(
    "Minimum (0.121)",
    xy=(400, 0.121),
    xytext=(550, 0.115),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#2c5f8a"),
    color="#2c5f8a",
    fontweight="bold",
)

# Annotate local peak
ax.annotate(
    "Local peak (0.133)",
    xy=(800, 0.133),
    xytext=(650, 0.139),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#999"),
    color="#666",
)

# Annotate final
ax.annotate(
    "Final (0.123)",
    xy=(1200, 0.123),
    xytext=(1050, 0.117),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#999"),
    color="#666",
)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Average Loss", fontsize=12)
ax.set_title("Hokusai LoRA Training Loss", fontsize=14)
ax.set_xticks(steps)
ax.set_ylim(0.110, 0.150)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output = "reports/hokusai/figures/loss_curve.png"
fig.savefig(output, dpi=300)
print(f"Saved: {output}")
