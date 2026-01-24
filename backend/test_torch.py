import sys
print("Importing torch...", flush=True)
try:
    import torch
    print(f"Torch imported successfully. Version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Error importing torch: {e}", flush=True)
