import numpy as np
import json
import os

base_dir = "backend"

try:
    emb_path = os.path.join(base_dir, "product_embeddings.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"Embeddings shape: {emb.shape}")
        if len(emb) > 0:
            print(f"Embeddings sample (first 5 of first item): {emb[0][:5]}")
            print(f"Are all zeros? {np.all(emb == 0)}")
            print(f"Any nans? {np.isnan(emb).any()}")
    else:
        print(f"{emb_path} not found")

    prod_path = os.path.join(base_dir, "products.json")
    if os.path.exists(prod_path):
        with open(prod_path, "r") as f:
            data = json.load(f)
        print(f"Products count: {len(data)}")
    else:
        print(f"{prod_path} not found")

except Exception as e:
    print(f"Error: {e}")
