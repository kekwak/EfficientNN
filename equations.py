import numpy as np


def flops(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return b * (17714 * s**2 + 313344)


def memory(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return 4_161_296 + 52 * b * s**2


def bytes_moved(image_size, batch):
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return 4 * (1_040_324 + b * (49 * s**2 + 2148))


def latency(image_size, batch, theta):
    f = flops(image_size, batch)
    d = bytes_moved(image_size, batch)
    return theta["t0"] + np.maximum(
        f / theta["p_eff"],
        d / theta["bw_eff"],
    )


def energy(image_size, batch, theta_energy):
    f = flops(image_size, batch)
    d = bytes_moved(image_size, batch)
    return (
        theta_energy["e0"]
        + theta_energy["alpha"] * f
        + theta_energy["beta"] * d
    )
