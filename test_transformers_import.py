
import sys
print("Starting import test...", flush=True)
try:
    import torch
    print(f"Torch version: {torch.__version__}", flush=True)
    from transformers import CLIPProcessor, CLIPModel
    print("Transformers imported successfully!", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)
print("Finished import test.", flush=True)
