import gc
import math
import random
import time
from pathlib import Path

import pandas as pd
import pynvml
import torch
from tqdm import tqdm

from models import model


SEED = 42
WARMUP = 10
REPEATS = 50

random.seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

model = model.cuda().eval()

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)


def measure_latency(x):
    for _ in range(WARMUP):
        model(x)

    torch.cuda.synchronize()
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return float(torch.tensor(times).median())


def measure_memory(x):
    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def measure_energy(x, latency):
    runs = max(2, math.ceil(1.0 / latency))

    torch.cuda.synchronize()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)

    for _ in range(runs):
        model(x)

    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)

    return (e1 - e0) / 1000 / runs


base_s = [32, 64, 128, 224, 256, 384, 512]
base_b = [1, 2, 4, 8, 16, 32, 64, 128, 256]

random_s = random.sample(
    [s for s in range(32, 513, 16) if s not in base_s],
    4,
)
random_b = random.sample(
    [b for b in range(1, 257) if b & (b - 1)],
    3,
)

sizes = base_s + random_s
batches = base_b + random_b

rows = []

with torch.inference_mode():
    for s in tqdm(sizes, desc="sizes"):
        for b in tqdm(batches, desc="batches"):
            validation = s in random_s or b in random_b
            x = None

            try:
                x = torch.randn(b, 3, s, s, device="cuda")
                measured_latency = measure_latency(x)

                rows.append({
                    "S": s,
                    "B": b,
                    "latency": measured_latency,
                    "memory": measure_memory(x),
                    "energy": measure_energy(x, measured_latency),
                    "is_validation": validation,
                    "oom": False,
                })

            except torch.cuda.OutOfMemoryError:
                rows.append({
                    "S": s,
                    "B": b,
                    "latency": None,
                    "memory": None,
                    "energy": None,
                    "is_validation": validation,
                    "oom": True,
                })

            finally:
                if x is not None:
                    del x
                gc.collect()
                torch.cuda.empty_cache()


Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/measurements.csv", index=False)
pynvml.nvmlShutdown()
