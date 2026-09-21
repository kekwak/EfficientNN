import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import energy, latency, memory


df = pd.read_csv("results/measurements.csv")
val = df[df["is_validation"] & ~df["oom"]].copy()

with open("results/theta.json") as f:
    theta = json.load(f)

val["latency_pred"] = latency(val["S"], val["B"], theta["latency"])
val["memory_pred"] = memory(val["S"], val["B"])
val["energy_pred"] = energy(val["S"], val["B"], theta["energy"])

Path("results/figures").mkdir(parents=True, exist_ok=True)

units = {
    "latency": (1e3, "ms"),
    "memory": (1 / 2**20, "MiB"),
    "energy": (1, "J"),
}

for name, (scale, unit) in units.items():
    data = val.dropna(subset=[name, f"{name}_pred"])
    measured = data[name] * scale
    predicted = data[f"{name}_pred"] * scale

    plt.figure()
    plt.scatter(measured, predicted)
    lo = min(measured.min(), predicted.min())
    hi = max(measured.max(), predicted.max())
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel(f"Measured {name} ({unit})")
    plt.ylabel(f"Predicted {name} ({unit})")
    plt.title(f"{name.capitalize()}: predicted vs measured")
    plt.tight_layout()
    plt.savefig(f"results/figures/{name}_parity.png", dpi=160)
    plt.close()

sizes = np.linspace(df["S"].min(), df["S"].max(), 60)
batches = np.linspace(df["B"].min(), df["B"].max(), 60)
S, B = np.meshgrid(sizes, batches)

surface_specs = {
    "latency": (latency(S, B, theta["latency"]) * 1e3, val["latency"] * 1e3, "ms"),
    "memory": (memory(S, B) / 2**20, val["memory"] / 2**20, "MiB"),
    "energy": (energy(S, B, theta["energy"]), val["energy"], "J"),
}

for name, (pred, measured, unit) in surface_specs.items():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(S, B, pred, alpha=0.45)
    ax.scatter(val["S"], val["B"], measured)
    ax.set_xlabel("Image size S (px)")
    ax.set_ylabel("Batch size B")
    ax.set_zlabel(f"{name.capitalize()} ({unit})")
    ax.set_title(f"{name.capitalize()}: predicted surface and measured points")
    fig.tight_layout()
    fig.savefig(f"results/figures/{name}_surface.png", dpi=160)
    plt.close(fig)

val.to_csv("results/validation.csv", index=False)
