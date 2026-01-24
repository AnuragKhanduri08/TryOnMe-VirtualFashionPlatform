import sys
print("Importing huggingface_hub...", flush=True)
try:
    import huggingface_hub
    print(f"Huggingface_hub imported successfully. Version: {huggingface_hub.__version__}", flush=True)
except Exception as e:
    print(f"Error importing huggingface_hub: {e}", flush=True)
