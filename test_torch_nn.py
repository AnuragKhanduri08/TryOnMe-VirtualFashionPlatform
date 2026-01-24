
import sys
print("Start", flush=True)
import torch
print("Torch imported", flush=True)
import torch.nn as nn
print("torch.nn imported", flush=True)
l = nn.Linear(10, 10)
print("Linear layer created", flush=True)
