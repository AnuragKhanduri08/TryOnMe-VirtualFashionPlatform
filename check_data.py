import numpy as np
import json
import os

try:
    print("Checking products.json...")
    with open("backend/products.json", "r") as f:
        products = json.load(f)
    print(f"Products count: {len(products)}")
except Exception as e:
    print(f"Error reading products.json: {e}")

try:
    print("Checking product_embeddings.npy...")
    embeddings = np.load("backend/product_embeddings.npy")
    print(f"Embeddings shape: {embeddings.shape}")
except Exception as e:
    print(f"Error reading product_embeddings.npy: {e}")
