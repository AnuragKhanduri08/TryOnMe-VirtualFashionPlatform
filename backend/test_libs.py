print("Importing torch...")
try:
    import torch
    print(f"Torch imported: {torch.__version__}")
except Exception as e:
    print(f"Torch fail: {e}")

print("Importing sentence_transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers imported")
except Exception as e:
    print(f"sentence_transformers fail: {e}")
