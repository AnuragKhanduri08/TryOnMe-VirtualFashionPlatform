import sys
print("Step 1: Importing transformers...", flush=True)
try:
    import transformers
    print(f"Transformers imported. Version: {transformers.__version__}", flush=True)
except Exception as e:
    print(f"Error importing transformers: {e}", flush=True)

print("Step 2: Importing sentence_transformers...", flush=True)
try:
    from sentence_transformers import SentenceTransformer
    print(f"SentenceTransformer imported.", flush=True)
except Exception as e:
    print(f"Error importing sentence_transformers: {e}", flush=True)

print("Step 3: Loading model...", flush=True)
try:
    # Use a tiny model or just check if class instantiates
    # model = SentenceTransformer('clip-ViT-B-32')
    print("Skipping model load for now, just checking imports.", flush=True)
except Exception as e:
    print(f"Error loading model: {e}", flush=True)
