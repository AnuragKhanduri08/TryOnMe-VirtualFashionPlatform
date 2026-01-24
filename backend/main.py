import os
import sys
import json
import time
import base64
import io
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
import torch

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AI Engines
try:
    print("Importing SmartSearchEngine...")
    from ai_modules.smart_search.engine import SmartSearchEngine
    print("SmartSearchEngine imported.")
except ImportError as e:
    SmartSearchEngine = None
    print(f"Warning: SmartSearchEngine not found: {e}")

try:
    print("Importing BodyMeasurementEstimator...")
    from ai_modules.body_measurement.estimator import BodyMeasurementEstimator
    print("BodyMeasurementEstimator imported.")
except ImportError as e:
    BodyMeasurementEstimator = None
    print(f"Warning: BodyMeasurementEstimator not found: {e}")

try:
    print("Importing VirtualTryOnEngine...")
    from ai_modules.virtual_try_on.engine import VirtualTryOnEngine
    print("VirtualTryOnEngine imported.")
except ImportError as e:
    VirtualTryOnEngine = None
    print(f"Warning: VirtualTryOnEngine not found: {e}")

# Global Variables
products = []
product_embeddings = None
product_histograms = None
search_engine = None
body_estimator = None
tryon_engine = None

metrics = {
    "total_requests": 0,
    "errors": 0,
    "latencies": [],
    "endpoint_usage": {},
    "recent_logs": []
}

# Load Data
def load_data():
    global products, product_embeddings, product_histograms
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    products_path = os.path.join(base_dir, "products.json")
    embeddings_path = os.path.join(base_dir, "product_embeddings.npy")
    histograms_path = os.path.join(base_dir, "product_histograms.npy")
    
    try:
        print("Loading products.json...")
        with open(products_path, "r") as f:
            products = json.load(f)
        print(f"Loaded {len(products)} products.")
    except Exception as e:
        print(f"Error loading products.json: {e}")
        products = []

    try:
        if os.path.exists(embeddings_path):
            print("Loading product_embeddings.npy...")
            product_embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings: {product_embeddings.shape}")
        else:
            print("product_embeddings.npy not found.")
            
        if os.path.exists(histograms_path):
            product_histograms = np.load(histograms_path)
            print(f"Loaded histograms: {product_histograms.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    load_data()
    
    global search_engine, body_estimator, tryon_engine
    
    # Initialize Search
    if SmartSearchEngine:
        try:
            print("Initializing Search Engine...")
            search_engine = SmartSearchEngine()
            print("Search Engine initialized.")
        except Exception as e:
            print(f"Failed to init Search Engine: {e}")

    # Initialize Body Measurement
    if BodyMeasurementEstimator:
        try:
            print("Initializing Body Measurement Estimator...")
            body_estimator = BodyMeasurementEstimator()
            print("Body Measurement Estimator initialized.")
        except Exception as e:
            print(f"Failed to init Body Estimator: {e}")

    # Initialize Try-On
    if VirtualTryOnEngine:
        try:
            print("Initializing Virtual Try-On Engine...")
            tryon_engine = VirtualTryOnEngine()
            print("Virtual Try-On Engine initialized.")
        except Exception as e:
            print(f"Failed to init Try-On Engine: {e}")
            
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["latencies"].append(process_time)
        if len(metrics["latencies"]) > 100:
            metrics["latencies"].pop(0)
            
        metrics["endpoint_usage"][endpoint] = metrics["endpoint_usage"].get(endpoint, 0) + 1
        
        return response
    except Exception as e:
        metrics["errors"] += 1
        metrics["recent_logs"].append(f"Error in {endpoint}: {str(e)}")
        if len(metrics["recent_logs"]) > 50:
             metrics["recent_logs"].pop(0)
        raise e

# Static Files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    os.makedirs(os.path.join(static_dir, "products"), exist_ok=True)
    os.makedirs(os.path.join(static_dir, "images"), exist_ok=True)

print(f"Mounting static directory: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def health_check():
    return {"status": "ok", "services": {
        "search": search_engine is not None,
        "body_measurement": body_estimator is not None,
        "virtual_try_on": tryon_engine is not None,
        "products_loaded": len(products)
    }}

@app.get("/products")
async def get_products(limit: int = 20):
    """
    Get a list of products (default catalog view).
    """
    if not products:
        raise HTTPException(status_code=503, detail="Product data not loaded")
    
    return products[:limit]

@app.get("/search")
async def search(
    q: str = Query(...), 
    limit: int = Query(20),
    use_ai: bool = Query(True)
):
    """
    Text search for products.
    """
    if not products:
        raise HTTPException(status_code=503, detail="Product data not loaded")
        
    results = []
    
    # 1. AI Search (CLIP)
    if use_ai and search_engine and search_engine.model is not None and product_embeddings is not None:
        try:
            # search_engine.search returns (values, indices)
            search_res = search_engine.search(q, product_embeddings, top_k=limit)
            indices = search_res[1]
            if hasattr(indices, 'cpu'):
                indices = indices.cpu().numpy()
            
            for idx in indices:
                idx_val = int(idx)
                if idx_val < len(products):
                    results.append(products[idx_val])
            return {"query": q, "results": results, "method": "ai"}
        except Exception as e:
            print(f"AI Search failed: {e}")
            # Fallback to keyword
            
    # 2. Keyword Fallback
    q_lower = q.lower()
    for p in products:
        if q_lower in p["name"].lower() or q_lower in p.get("articleType", "").lower():
            results.append(p)
            if len(results) >= limit:
                break
                
    return {"query": q, "results": results, "method": "keyword"}

@app.get("/search/suggestions")
def get_suggestions(q: str, limit: int = 5):
    if not products:
        return {"query": q, "suggestions": []}
        
    q_lower = q.lower()
    suggestions = []
    seen = set()
    
    # Simple substring match
    for p in products:
        name = p["name"]
        if q_lower in name.lower():
            if name not in seen:
                suggestions.append(name)
                seen.add(name)
            if len(suggestions) >= limit:
                break
                
    return {"query": q, "suggestions": suggestions}

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), limit: int = 20):
    """
    Semantic search using an image (Image-to-Text).
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform search
        # Pass product_histograms if available (global variable)
        ph = product_histograms if 'product_histograms' in globals() else None
        results = search_engine.search(image, product_embeddings, product_histograms=ph, top_k=limit)
        
        search_results = []
        
        if search_engine.model is None:
            # Histogram search returns list of (pid, score)
            for pid, score in results:
                prod = next((p for p in products if p["id"] == pid), None)
                if prod:
                    # Add score to result for debugging
                    prod_copy = prod.copy()
                    prod_copy["score"] = float(score)
                    search_results.append(prod_copy)
        else:
            # Embedding search returns (values, indices)
            for idx in results[1]:
                search_results.append(products[idx.item()])
            
        return {"query": "image_upload", "results": search_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/measure")
async def measure_body(file: UploadFile = File(...)):
    """
    Estimate body measurements from an uploaded image.
    """
    if not body_estimator:
        raise HTTPException(status_code=503, detail="Body measurement estimator not initialized")
    
    try:
        # Read image
        contents = await file.read()
        
        # Convert to numpy array (BGR for OpenCV/YOLO)
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_np is None:
             raise HTTPException(status_code=400, detail="Invalid image data")

        # Perform estimation
        results = body_estimator.estimate_from_image(image_np)
        
        if results.get("status") == "error":
            # We can decide to return 200 with error info or 400. 
            # Let's return 200 but with status: error so frontend can handle it gracefully
            pass

        return results
        
    except Exception as e:
        print(f"Error in measure_body: {e}")
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")

@app.get("/recommend/{product_id}")
def recommend_products(product_id: int, limit: int = 20):
    """
    Recommend similar products based on metadata and 'Complete the Look' matching.
    """
    # Find product
    source_product = next((p for p in products if p["id"] == product_id), None)
            
    if not source_product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    # --- 1. Find Similar Items ---
    similar_items = []
    
    # Try Semantic Similarity first (if available)
    if search_engine and search_engine.model is not None and product_embeddings is not None:
        try:
            # Find index of source product
            # Assuming product_embeddings aligns exactly with products list
            source_idx = next((i for i, p in enumerate(products) if p["id"] == product_id), None)
            
            if source_idx is not None:
                source_embedding = product_embeddings[source_idx]
                # Search
                results = search_engine.search_by_embedding(source_embedding, product_embeddings, top_k=limit+1)
                
                # Extract products (excluding self)
                # results is (values, indices)
                # Ensure we handle torch tensor or numpy array output from search_by_embedding
                indices = results[1]
                if hasattr(indices, 'cpu'):
                    indices = indices.cpu().numpy()
                
                for idx in indices:
                    idx_val = int(idx) # Ensure integer
                    if idx_val != source_idx and idx_val < len(products):
                        similar_items.append(products[idx_val])
                
                # Limit
                similar_items = similar_items[:limit]
        except Exception as e:
            print(f"Error in semantic recommendation: {e}")
            similar_items = []

    # Fallback to Metadata Similarity if Semantic failed or returned nothing
    if not similar_items:
        # Strategy: Same Gender, Same Category, Same/Similar Color
        candidates = [p for p in products if p["id"] != product_id and p.get("gender") == source_product.get("gender")]
        
        # Scoring for Similarity
        scored_similar = []
        for p in candidates:
            score = 0
            # Category Match (High Priority)
            if p.get("articleType") == source_product.get("articleType"):
                score += 20
            elif p.get("subCategory") == source_product.get("subCategory"):
                score += 10
                
            # Color Match
            if p.get("baseColour") == source_product.get("baseColour"):
                score += 15
                
            # Usage/Season Match
            if p.get("usage") == source_product.get("usage"):
                score += 5
            if p.get("season") == source_product.get("season"):
                score += 5
                
            scored_similar.append((score, p))
            
        # Sort by score desc
        scored_similar.sort(key=lambda x: x[0], reverse=True)
        similar_items = [x[1] for x in scored_similar[:limit]]
    
    # --- 2. Find Matching Items (Outfit Completion) ---
    # Strategy: Same Gender, Complementary Category, Compatible Color
    
    candidates = [p for p in products if p["id"] != product_id and p.get("gender") == source_product.get("gender")]
    
    source_sub = source_product.get("subCategory")
    target_subs = []
    
    if source_sub == "Topwear":
        target_subs = ["Bottomwear", "Shoes"]
    elif source_sub == "Bottomwear":
        target_subs = ["Topwear", "Shoes"]
    elif source_sub == "Shoes":
        target_subs = ["Topwear", "Bottomwear"]
    else:
        # Fallback for accessories etc. -> match with Apparel
        target_subs = ["Topwear", "Bottomwear"]
        
    scored_matches = []
    
    # Neutral colors that match with everything
    neutrals = ["Black", "White", "Grey", "Navy Blue", "Beige", "Silver"]
    
    for p in candidates:
        if p.get("subCategory") not in target_subs:
            continue
            
        score = 0
        
        # Context Match
        if p.get("usage") == source_product.get("usage"):
            score += 10
        if p.get("season") == source_product.get("season"):
            score += 10
            
        # Color Matching Logic
        p_color = p.get("baseColour")
        s_color = source_product.get("baseColour")
        
        # If source is neutral, almost anything matches
        if s_color in neutrals:
            score += 5
        # If candidate is neutral, it likely matches source
        if p_color in neutrals:
            score += 5
            
        # If both are same color (monochromatic look)
        if p_color == s_color:
            score += 3
            
        scored_matches.append((score, p))
        
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Try to pick a mix of categories if possible (e.g. 1 Bottom, 1 Shoe)
    matching_items = []
    seen_subs = set()
    
    # First pass: try to get distinct subcategories
    for score, p in scored_matches:
        if p.get("subCategory") not in seen_subs:
            matching_items.append(p)
            seen_subs.add(p.get("subCategory"))
            
    # Fill up the rest
    for score, p in scored_matches:
        if p not in matching_items:
            matching_items.append(p)
            
    matching_items = matching_items[:limit]
                
    return {
        "source_product_id": product_id, 
        "recommendations": similar_items,
        "matching_items": matching_items
    }

@app.get("/dashboard/stats")
def get_dashboard_stats():
    """
    Get system monitoring metrics.
    """
    avg_latency = 0
    if metrics["latencies"]:
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])
        
    return {
        "total_requests": metrics["total_requests"],
        "avg_latency": avg_latency,
        "errors": metrics["errors"],
        "error_rate": (metrics["errors"] / metrics["total_requests"]) if metrics["total_requests"] > 0 else 0,
        "endpoint_usage": metrics["endpoint_usage"],
        "recent_logs": metrics["recent_logs"]
    }

@app.post("/segment")
async def segment_image(
    file: Optional[UploadFile] = File(None),
    cloth_id: Optional[str] = Form(None)
):
    """
    Remove background from an image.
    """
    if not tryon_engine:
         raise HTTPException(status_code=503, detail="Try-On engine not initialized")
    
    try:
        image_np = None
        
        if cloth_id:
             cloth_path = os.path.join(static_dir, "products", f"{cloth_id}.jpg")
             if os.path.exists(cloth_path):
                 image_np = cv2.imread(cloth_path, cv2.IMREAD_COLOR)
             else:
                 raise HTTPException(status_code=404, detail="Product not found")
        elif file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if image_np is None:
             raise HTTPException(status_code=400, detail="Invalid image")
             
        # Use default "upper" cloth type or infer? 
        # The user requested specific segmentation for shirts in manual mode (which uses try-on),
        # but this is the /segment endpoint. Let's default to "upper" as it's the most common use case.
        segmented = tryon_engine.segment_cloth(image_np, cloth_type="upper")
        
        # Return as PNG
        success, encoded_image = cv2.imencode('.png', segmented)
        if not success:
             raise HTTPException(status_code=500, detail="Encoding failed")
             
        return Response(content=encoded_image.tobytes(), media_type="image/png")
    except Exception as e:
        print(f"Segment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/try-on")
async def virtual_try_on(
    person_image: UploadFile = File(...), 
    cloth_image: Optional[UploadFile] = File(None),
    cloth_id: Optional[str] = Form(None),
    adj_scale: float = Form(1.0),
    adj_x: float = Form(0.0),
    adj_y: float = Form(0.0),
    use_cloud: str = Form("false"),
    hf_token: Optional[str] = Form(None)
):
    """
    Virtual Try-On Endpoint.
    """
    # Debug Print
    print(f"Try-On Request: use_cloud_raw={use_cloud}, type={type(use_cloud)}")
    
    # Explicit boolean conversion
    use_cloud_bool = str(use_cloud).lower() == 'true'
    print(f"Try-On Request: use_cloud_bool={use_cloud_bool}")

    if not tryon_engine or not body_estimator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Read Person Image
        person_bytes = await person_image.read()
        nparr_person = np.frombuffer(person_bytes, np.uint8)
        person_np = cv2.imdecode(nparr_person, cv2.IMREAD_COLOR)
        
        # Read Cloth Image
        cloth_np = None
        
        if cloth_id:
            # Load from static files
            # Assuming IDs map to filenames in static/products
            cloth_path = os.path.join(static_dir, "products", f"{cloth_id}.jpg")
            if not os.path.exists(cloth_path):
                 # Try finding it in the products list to see if there's a custom URL logic (unlikely based on context)
                 # But let's fallback to checking if it exists
                 raise HTTPException(status_code=404, detail=f"Product image for ID {cloth_id} not found")
            
            cloth_np = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)
            if cloth_np is None:
                raise HTTPException(status_code=400, detail="Failed to load product image from server")
                
        elif cloth_image:
            cloth_bytes = await cloth_image.read()
            nparr_cloth = np.frombuffer(cloth_bytes, np.uint8)
            # Use IMREAD_UNCHANGED to preserve alpha channel if present
            cloth_np = cv2.imdecode(nparr_cloth, cv2.IMREAD_UNCHANGED)
            
        else:
             raise HTTPException(status_code=400, detail="Please provide either a cloth image or a cloth_id")
        
        if person_np is None or cloth_np is None:
             raise HTTPException(status_code=400, detail="Invalid image data")

        # Get Pose Keypoints (Required even for Cloud as a fallback check or for future hybrid logic)
        pose_result = body_estimator.estimate_from_image(person_np)
        
        # If pose detection fails, we might still want to try cloud if enabled, 
        # but usually pose failure means bad image.
        keypoints = []
        if pose_result["status"] == "success":
             keypoints = pose_result["keypoints"]
        
        # Run Try-On with Adjustments
        adjustments = {
            "scale": adj_scale,
            "x": adj_x,
            "y": adj_y
        }
        
        result_np = tryon_engine.try_on(person_np, cloth_np, keypoints, adjustments=adjustments, use_cloud=use_cloud_bool, hf_token=hf_token)
        
        if result_np is None:
             raise HTTPException(status_code=500, detail="Try-On generated no result.")

        # Encode result to base64
        _, buffer = cv2.imencode('.jpg', result_np)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "result_image": f"data:image/jpeg;base64,{img_str}"
        }
        
    except Exception as e:
        print(f"Error in try-on: {e}")
        raise HTTPException(status_code=500, detail=f"Virtual Try-On failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Uvicorn failed: {e}")
    print("Uvicorn exited.")
