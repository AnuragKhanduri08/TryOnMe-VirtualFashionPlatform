
print("Starting...", flush=True)
try:
    from transformers import CLIPProcessor
    print("transformers imported", flush=True)
except Exception as e:
    print(f"transformers fail: {e}", flush=True)

import torch
print("torch imported", flush=True)
