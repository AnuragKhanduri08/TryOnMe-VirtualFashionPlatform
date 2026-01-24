
import os
import sys
import json
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_modules.smart_search.engine import SmartSearchEngine

def test_subset():
    print("Testing subset precomputation...")
    base_dir = os.path.join(os.getcwd(), "backend")
    products_path = os.path.join(base_dir, "products.json")
    
    with open(products_path, "r") as f:
        products = json.load(f)
        
    subset = products[:100]
    descriptions = [p["description"] for p in subset]
    
    engine = SmartSearchEngine()
    
    start = time.time()
    embeddings = engine.encode_text(descriptions, batch_size=32)
    end = time.time()
    
    print(f"Computed 100 embeddings in {end - start:.2f}s", flush=True)
    print(f"Shape: {embeddings.shape}", flush=True)
    
    if np.all(embeddings.numpy() == 0):
        print("ERROR: Embeddings are all zeros!", flush=True)
    else:
        print("SUCCESS: Embeddings contain non-zero values.", flush=True)

if __name__ == "__main__":
    test_subset()
