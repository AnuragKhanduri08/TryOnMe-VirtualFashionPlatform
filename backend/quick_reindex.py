import json
import os
import sys
import numpy as np
import time

print("Script started...", flush=True)

# Add the project root to sys.path so we can import from ai_modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing SmartSearchEngine...", flush=True)
from ai_modules.smart_search.engine import SmartSearchEngine
print("Imported SmartSearchEngine.", flush=True)

def quick_reindex(limit=2000):
    products_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products.json")
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "product_embeddings.npy")
    
    if not os.path.exists(products_path):
        print(f"Error: {products_path} not found.")
        return

    print(f"Loading products from {products_path}...")
    with open(products_path, 'r') as f:
        products = json.load(f)
    
    print(f"Total products found: {len(products)}")
    
    # Slice to limit
    if len(products) > limit:
        print(f"Limiting to first {limit} products for quick re-indexing...", flush=True)
        products = products[:limit]
        
        # Save back to products.json to ensure consistency
        print("Saving truncated products.json...", flush=True)
        with open(products_path, 'w') as f:
            json.dump(products, f, indent=2)
    
    # Initialize Engine
    print("Initializing Smart Search Engine...", flush=True)
    try:
        engine = SmartSearchEngine()
    except Exception as e:
        print(f"Failed to init engine: {e}", flush=True)
        return
    
    if engine.model is None:
        print("Error: CLIP model not loaded. Cannot generate embeddings.", flush=True)
        return

    # Generate Embeddings
    print("Generating embeddings...", flush=True)
    embeddings = []
    
    start_time = time.time()
    batch_size = 32
    
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        texts = [p.get('description', p.get('name', '')) for p in batch]
        
        try:
            # SmartSearchEngine.encode_text returns a tensor
            batch_embeddings = engine.encode_text(texts)
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(products)} products...")
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Append zeros as fallback to maintain shape
            embeddings.append(np.zeros((len(batch), 512)))

    # Concatenate all batches
    if embeddings:
        final_embeddings = np.concatenate(embeddings, axis=0)
        print(f"Final embeddings shape: {final_embeddings.shape}")
        
        # Save
        print(f"Saving embeddings to {embeddings_path}...")
        np.save(embeddings_path, final_embeddings)
        print("Re-indexing complete!")
    else:
        print("No embeddings generated.")

    print(f"Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    quick_reindex()
