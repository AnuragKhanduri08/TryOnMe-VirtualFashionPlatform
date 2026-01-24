import sys
print("Importing sentence_transformers...", flush=True)
try:
    from sentence_transformers import SentenceTransformer
    print("Import successful.", flush=True)
    print("Loading model clip-ViT-B-32...", flush=True)
    model = SentenceTransformer('clip-ViT-B-32')
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
