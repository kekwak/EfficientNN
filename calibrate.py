import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import flops, bytes_moved, latency, energy


df = pd.read_csv("results/measurements.csv")
train = df[(~df["is_validation"]) & (~df["oom"])]


def latency_residuals(x):
    theta = {
        "t0": np.exp(x[0]),
        "p_eff": np.exp(x[1]),
        "bw_eff": np.exp(x[2]),
    }

    pred = latency(train["S"], train["B"], theta)
    return pred - train["latency"].to_numpy()


x = least_squares(
    latency_residuals,
    np.log([1e-4, 1e12, 1e11]),
).x

theta_latency = {
    "t0": np.exp(x[0]),
    "p_eff": np.exp(x[1]),
    "bw_eff": np.exp(x[2]),
}


energy_train = train.dropna(subset=["energy"])
energy_train = energy_train[energy_train["energy"] > 0]

f = flops(energy_train["S"], energy_train["B"])
d = bytes_moved(energy_train["S"], energy_train["B"])
X = np.column_stack([np.ones(len(energy_train)), f, d])

e = least_squares(
    lambda x: X @ x - energy_train["energy"].to_numpy(),
    x0=[1e-3, 1e-12, 1e-10],
    bounds=(0, np.inf),
).x

theta_energy = {
    "e0": e[0],
    "alpha": e[1],
    "beta": e[2],
}


theta = {
    "latency": theta_latency,
    "energy": theta_energy,
}

with open("results/theta.json", "w") as f:
    json.dump(theta, f, indent=2)

print(theta)
