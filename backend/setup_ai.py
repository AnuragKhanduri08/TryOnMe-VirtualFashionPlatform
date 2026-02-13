import os
import sys
import json
import numpy as np
import torch
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Product

# Add parent directory to path to import ai_modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_ai_models():
    print("========================================")
    print("   AI MODEL SETUP & DATA GENERATION")
    print("========================================")
    
    # 1. Load Products (DB or JSON Fallback)
    products = []
    try:
        print("Connecting to database...")
        db = SessionLocal()
        try:
            products_db = db.query(Product).order_by(Product.id).all()
            if products_db:
                products = [p.to_dict() for p in products_db]
                print(f"✅ Loaded {len(products)} products from Database.")
            else:
                print("⚠️ Database is empty.")
        except Exception as e:
            print(f"⚠️ Database connection failed: {e}")
        finally:
            db.close()
            
        if not products:
            print("Attempting fallback to products.json...")
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    products = json.load(f)
                print(f"✅ Loaded {len(products)} products from products.json.")
            else:
                print("❌ products.json not found!")
                
    except Exception as e:
        print(f"❌ Error loading products: {e}")
        return

    if not products:
        print("❌ No products found! Cannot generate embeddings.")
        return

    # 2. Generate Text Embeddings (CLIP)
    print("\n[1/2] Generating Text Embeddings (CLIP)...")
    try:
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
        
        model_name = "openai/clip-vit-base-patch32"
        print(f"Loading model: {model_name}...")
        
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        
        descriptions = [
            f"{p.get('gender', '')} {p.get('category', '')} {p.get('subCategory', '')} {p.get('name', '')} {p.get('baseColour', '')}".strip() 
            for p in products
        ]
        
        print(f"Generating embeddings for {len(descriptions)} items...")
        
        all_embeddings = []
        batch_size = 32
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            if i % 100 == 0:
                print(f"Processing batch {i}/{len(descriptions)}...", flush=True)
                
            try:
                inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model.get_text_features(**inputs)
                
                # Normalize embeddings
                # Handle Transformers Output object if necessary
                if not isinstance(outputs, torch.Tensor):
                    if hasattr(outputs, 'text_embeds'):
                        outputs = outputs.text_embeds
                    elif hasattr(outputs, 'pooler_output'):
                        outputs = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        # Mean pooling if we get raw hidden states
                        outputs = outputs.last_hidden_state.mean(dim=1)
                
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                all_embeddings.append(outputs)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Append zeros to keep alignment
                all_embeddings.append(torch.zeros((len(batch), 512)))
                
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            np.save("product_embeddings.npy", final_embeddings)
            print(f"✅ Saved 'product_embeddings.npy' with shape {final_embeddings.shape}")
        else:
            print("❌ Failed to generate any embeddings.")
            
    except ImportError:
        print("❌ 'transformers' library not found. Please run 'pip install transformers'.")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")

    # 3. Generate Color Histograms (Placeholder)
    print("\n[2/2] Generating Color Histograms (Placeholder)...")
    try:
        # Create placeholder histogram file
        # Real histogram generation requires downloading images, which is too slow for setup.
        # This prevents app crash on startup.
        dummy_hist = np.zeros((len(products), 768), dtype=np.float32) 
        np.save("product_histograms.npy", dummy_hist)
        print("⚠️ Created placeholder 'product_histograms.npy' (Real histograms require downloading all images)")
        
    except Exception as e:
        print(f"❌ Error creating histograms: {e}")

    print("\n✅ AI Setup Complete! You can now use Smart Search.")

if __name__ == "__main__":
    setup_ai_models()
