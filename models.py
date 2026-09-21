import torch as th
import torch.nn as nn

# | Layers | Output resolution |
# |--------|-------------------|
# | Conv7×7 s2 3→32, MaxPool 3×3 s2 p1 | S/4 |
# | Conv5×5 32→64 | S/4 |
# | Conv3×3 s2 64→128 | S/8 |
# | Conv1×1 128→256 | S/8 |
# | Conv3×3 s2 256→256 | S/16 |
# | Conv1×1 256→512 | S/16 |
# | head: GlobalAvgPool, Linear 512→256, ReLU, Linear 256→100 | — |

model = nn.Sequential(
    nn.Conv2d(3, 32, 7, 2, 7//2, bias=False),
    nn.MaxPool2d(3, 2, 1),
    nn.Conv2d(32, 64, 5, 1, 5//2, bias=False),
    nn.Conv2d(64, 128, 3, 2, 3//2, bias=False),
    nn.Conv2d(128, 256, 1, 1, 1//2, bias=False),
    nn.Conv2d(256, 256, 3, 2, 3//2, bias=False),
    nn.Conv2d(256, 512, 1, 1, 1//2, bias=False),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 100),
)
