import torch
print("torch imported", flush=True)

# Import transformers
print("Importing transformers...", flush=True)
try:
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    print("transformers imported (CLIP classes)", flush=True)
    _TRANSFORMERS_AVAILABLE = True
except BaseException as e:
    print(f"transformers import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    _TRANSFORMERS_AVAILABLE = False

import numpy as np
print("numpy imported", flush=True)

import cv2
print("cv2 imported", flush=True)

from typing import List, Union
from PIL import Image

class SmartSearchEngine:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        """
        Initializes the CLIP model (via transformers) or Fallback Histogram Engine.
        """
        self.model = None
        self.processor = None
        self.histograms = {} # Stores {product_id: histogram}
        
        if not _TRANSFORMERS_AVAILABLE:
            print("Smart Search Engine running in Fallback Mode (Color Histograms).")
            # Don't return here, allows usage of compute_histogram
        
        if _TRANSFORMERS_AVAILABLE:
            print(f"Loading Smart Search Model: {model_name}...")
            try:
                print("Loading CLIPProcessor...", flush=True)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                print("CLIPProcessor loaded.", flush=True)
                print("Loading CLIPModel...", flush=True)
                self.model = CLIPModel.from_pretrained(model_name)
                print("CLIPModel loaded.", flush=True)
                self.model.eval() # Set to evaluation mode
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Failed to load model: {e}", flush=True)
                self.model = None
                self.processor = None

    def compute_histogram(self, image: Union[Image.Image, np.ndarray]):
        """Computes HSV color histogram for an image."""
        import cv2 # Lazy import to avoid DLL hell
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if image is None:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram (Hue, Saturation) - ignore Value to be robust to lighting
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        
        # Normalize
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist.flatten()

    def encode_text(self, text: Union[str, List[str]], batch_size: int = 32):
        """Encodes text into a vector with batching."""
        if self.model is None:
            # Mock
            count = 1 if isinstance(text, str) else len(text)
            return torch.zeros((count, 512))
            
        if isinstance(text, str):
            text = [text]
            
        all_embeddings = []
        
        for i in range(0, len(text), batch_size):
            if i % (batch_size * 50) == 0:
                print(f"Processing batch starting at {i}/{len(text)}...", flush=True)
            batch = text[i:i + batch_size]
            try:
                print(f"Batch {i}: Encoding...", flush=True)
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                
                # Normalize embeddings
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                all_embeddings.append(outputs)
            except Exception as e:
                print(f"Error encoding batch {i}: {e}")
                # Append zeros for failed batch to keep alignment
                all_embeddings.append(torch.zeros((len(batch), 512)))

        if not all_embeddings:
            return torch.zeros((0, 512))
            
        return torch.cat(all_embeddings, dim=0)

    def encode_image(self, image: Union[Image.Image, List[Image.Image]]):
        """Encodes an image into a vector."""
        if self.model is None:
             # Return mock embedding
             count = 1 if isinstance(image, Image.Image) else len(image)
             return torch.zeros((count, 512))
             
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Normalize embeddings
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs

    def search(self, query: Union[str, Image.Image], product_embeddings=None, product_histograms=None, top_k=5):
        """
        Searches for products that match the query (text or image).
        """
        if isinstance(query, str):
            if self.model is None:
                # Fallback: Cannot do semantic search without model.
                # Caller should handle keyword search.
                return [] 
            query_embedding = self.encode_text(query)
            return self.search_by_embedding(query_embedding, product_embeddings, top_k)
        
        elif isinstance(query, (Image.Image, np.ndarray)):
            # Image Search
            if self.model is None:
                # Use Histogram Search
                return self.search_by_histogram(query, product_histograms, top_k)
            else:
                if isinstance(query, np.ndarray):
                    query = Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
                query_embedding = self.encode_image(query)
                return self.search_by_embedding(query_embedding, product_embeddings, top_k)
        
        return []

    def search_by_embedding(self, query_embedding, product_embeddings, top_k=5):
        """
        Searches using a pre-computed query embedding.
        """
        if product_embeddings is None:
             return []


        # Convert to torch tensor if numpy
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)
        if isinstance(product_embeddings, np.ndarray):
            product_embeddings = torch.from_numpy(product_embeddings)
            
        # Ensure float32
        query_embedding = query_embedding.float()
        product_embeddings = product_embeddings.float()

        # Ensure 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Compute cosine similarity
        # from sentence_transformers import util
        # cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
        # We can't use util if not imported. use torch manually.
        
        # Normalize
        a_norm_val = query_embedding.norm(dim=1)[:, None]
        b_norm_val = product_embeddings.norm(dim=1)[:, None]
        
        # Avoid division by zero
        a_norm_val[a_norm_val == 0] = 1e-9
        b_norm_val[b_norm_val == 0] = 1e-9
        
        a_norm = query_embedding / a_norm_val
        b_norm = product_embeddings / b_norm_val
        
        # Matrix multiplication
        # print(f"Computing similarity: {a_norm.shape} x {b_norm.shape}.T")
        cos_scores = torch.mm(a_norm, b_norm.transpose(0, 1))[0]
        
        # Sort results
        top_results = torch.topk(cos_scores, k=min(top_k, len(product_embeddings)))
        
        return top_results

    def search_by_histogram(self, query_image, product_histograms, top_k=5):
        """
        Searches using Color Histogram Comparison (Correlation).
        product_histograms: dict {product_id: histogram_array}
        """
        if not product_histograms:
            return []

        query_hist = self.compute_histogram(query_image)
        if query_hist is None:
            return []

        scores = []
        ids = []
        
        for pid, p_hist in product_histograms.items():
            if p_hist is None:
                continue
            # Compare Histograms (Correlation: Higher is better)
            score = cv2.compareHist(query_hist, p_hist, cv2.HISTCMP_CORREL)
            scores.append(score)
            ids.append(pid)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]
        
        # Return format similar to torch.topk (values, indices) but we return IDs here
        # Actually caller expects indices or objects. 
        # Let's return list of (pid, score)
        
        results = []
        for i in top_indices:
            results.append((ids[i], scores[i]))
            
        return results
