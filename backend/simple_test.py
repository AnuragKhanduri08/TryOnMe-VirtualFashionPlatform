print("Hello from simple test")
import sys
print(f"Python version: {sys.version}")
try:
    import torch
    print("Torch imported")
except ImportError:
    print("Torch failed")

print("Importing sentence_transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers imported")
except ImportError:
    print("sentence_transformers failed")
except Exception as e:
    print(f"sentence_transformers crash: {e}")
