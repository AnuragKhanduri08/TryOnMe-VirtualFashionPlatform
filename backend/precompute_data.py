
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import pickle
import time
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Engine FIRST (imports torch/transformers)
from ai_modules.smart_search.engine import SmartSearchEngine

import numpy as np
import cv2

def precompute():
    print("Starting pre-computation of embeddings and histograms...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    products_path = os.path.join(base_dir, "products.json")
    embeddings_path = os.path.join(base_dir, "product_embeddings.npy")
    histograms_path = os.path.join(base_dir, "product_histograms.pkl")
    static_dir = os.path.join(base_dir, "static")
    
    # Load Products
    if not os.path.exists(products_path):
        print("products.json not found!")
        return

    with open(products_path, "r") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products.")

    # Initialize Engine
    try:
        engine = SmartSearchEngine()
    except Exception as e:
        with open("precompute.log", "a") as log:
            log.write(f"Engine init failed: {e}\n")
        print(f"Engine init failed: {e}")
        return
    
    # 1. Compute Text Embeddings (if needed)
    # Check if we need to recompute (mismatch or missing)
    recompute_embeddings = True
    if os.path.exists(embeddings_path):
        try:
            existing = np.load(embeddings_path)
            if len(existing) == len(products):
                print("Embeddings already exist and match count. Skipping.")
                recompute_embeddings = False
            else:
                print(f"Embeddings count mismatch ({len(existing)} vs {len(products)}). Recomputing...")
        except:
            print("Error loading existing embeddings. Recomputing...")
            
    if recompute_embeddings and engine.model is not None:
        print("Computing text embeddings...", flush=True)
        with open("precompute.log", "a") as log:
            log.write("Starting text embeddings computation...\n")
            
        # Chunked processing
        descriptions = [p["description"] for p in products]
        total = len(descriptions)
        chunk_size = 1000
        
        all_embeddings_list = []
        
        # Check if we have partial embeddings (optional, for now start fresh)
        # If we wanted to resume, we'd load existing and start from len(existing)
        
        start_t = time.time()
        try:
            for i in range(0, total, chunk_size):
                print(f"Processing chunk {i}/{total}...", flush=True)
                chunk = descriptions[i : i + chunk_size]
                
                # Encode chunk
                chunk_embeddings = engine.encode_text(chunk, batch_size=32)
                all_embeddings_list.append(chunk_embeddings)
                
                # Optional: Save intermediate result?
                # It's better to just keep in memory if 90MB. 
                # But if it crashes, we lose it.
                # Let's try to finish first.
                
            # Concatenate
            if all_embeddings_list:
                embeddings = torch.cat(all_embeddings_list, dim=0)
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                end_t = time.time()
                print(f"Computed embeddings in {end_t - start_t:.2f}s", flush=True)
                
                np.save(embeddings_path, embeddings_np)
                print("Saved embeddings.", flush=True)
                with open("precompute.log", "a") as log:
                    log.write(f"Embeddings saved. Time: {end_t - start_t:.2f}s\n")
            else:
                print("No embeddings computed.")

        except Exception as e:
            print(f"Error computing embeddings: {e}", flush=True)
            import traceback
            traceback.print_exc()
            with open("precompute.log", "a") as log:
                log.write(f"Error computing embeddings: {e}\n")
            
    elif engine.model is None:
        print("Transformers not available. Skipping text embeddings.")
        with open("precompute.log", "a") as log:
            log.write("Transformers not available.\n")

    # 2. Compute Histograms
    # We want to support all images
    product_histograms = {}
    if os.path.exists(histograms_path):
        try:
            with open(histograms_path, "rb") as f:
                product_histograms = pickle.load(f)
            print(f"Loaded {len(product_histograms)} histograms.")
        except:
            product_histograms = {}
            
    print("Computing missing histograms...")
    count = 0
    updated = False
    
    # We will try to process ALL products, but we need to check if we have the images locally
    # Since the user switched to CLOUD images (Kaggle URLs), we might NOT have local images for histogram computation
    # unless we download them. 
    # BUT, the histogram logic in main.py looks for /static/ files.
    # If image_url is remote, we can't compute histogram easily without downloading.
    # Let's check how many local images we actually have.
    
    local_images_count = 0
    for p in tqdm(products):
        pid = p["id"]
        if pid in product_histograms:
            continue
            
        # Check if we have a local file
        # Even if image_url is remote, we might have the file in static/products/{id}.jpg
        local_path = os.path.join(static_dir, "products", f"{pid}.jpg")
        
        if os.path.exists(local_path):
            try:
                img = cv2.imread(local_path)
                if img is not None:
                    hist = engine.compute_histogram(img)
                    product_histograms[pid] = hist
                    updated = True
                    count += 1
            except Exception as e:
                pass
        else:
            # If no local image, we skip histogram for now (can't do image search without image data)
            pass
            
    print(f"Computed {count} new histograms. Total: {len(product_histograms)}")
    
    if updated:
        with open(histograms_path, "wb") as f:
            pickle.dump(product_histograms, f)
        print("Saved histograms.")

if __name__ == "__main__":
    precompute()
