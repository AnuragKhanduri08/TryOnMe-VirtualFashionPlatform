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
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Database Imports
from database import SessionLocal, engine
from models import Base, Product

# Create tables if not exist (mostly for SQLite fallback, migration script handles this usually)
Base.metadata.create_all(bind=engine)

from sqlalchemy import or_, func, desc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global Variables
product_ids = [] # Only store IDs to map embeddings index -> DB ID
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
    global product_ids, product_embeddings, product_histograms
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(base_dir, "product_embeddings.npy")
    histograms_path = os.path.join(base_dir, "product_histograms.npy")
    
    print("--- START DATA LOADING ---")
    
    # 1. Load Product IDs from DB (Lightweight)
    try:
        print("Loading product IDs from Database...")
        db = SessionLocal()
        try:
            # We MUST order by ID to ensure alignment with embeddings if they were generated in ID order
            ids_query = db.query(Product.id).order_by(Product.id).all()
            product_ids = [r[0] for r in ids_query]
            print(f"✅ Loaded {len(product_ids)} product IDs from Database.")
            
            if not product_ids:
                print("⚠️ Database returned 0 products.")
        except Exception as e:
            print(f"❌ DB Error loading IDs: {e}")
            product_ids = []
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Critical Error loading data: {e}")
        product_ids = []
        
    print(f"📊 Final Product ID Count in Memory: {len(product_ids)}")

    # 2. Load Embeddings (Numpy - Mapped to product_ids by index)
    try:
        if os.path.exists(embeddings_path):
            print("Loading product_embeddings.npy...")
            loaded_embeddings = np.load(embeddings_path)
            
            # Ensure alignment
            if len(loaded_embeddings) == len(product_ids):
                product_embeddings = loaded_embeddings
                print(f"✅ Loaded embeddings: {product_embeddings.shape} (Aligned with IDs)")
            else:
                print(f"⚠️ Embeddings count ({len(loaded_embeddings)}) != ID count ({len(product_ids)}). Truncating to min.")
                min_len = min(len(loaded_embeddings), len(product_ids))
                product_embeddings = loaded_embeddings[:min_len]
                product_ids = product_ids[:min_len]
                
        else:
            print("product_embeddings.npy not found.")
            
        if os.path.exists(histograms_path):
            loaded_histograms = np.load(histograms_path)
            if len(loaded_histograms) == len(product_ids):
                product_histograms = loaded_histograms
                print(f"✅ Loaded histograms: {product_histograms.shape}")
            else:
                 # Align
                 min_len = min(len(loaded_histograms), len(product_ids))
                 product_histograms = loaded_histograms[:min_len]
                 # If embeddings were longer, we might have issue, but usually generated together
    except Exception as e:
        print(f"Error loading numpy data: {e}")

# Helper: Lazy Loaders
def get_search_engine():
    global search_engine
    if search_engine is None:
        try:
            print("Lazy Loading Search Engine...")
            from ai_modules.smart_search.engine import SmartSearchEngine
            search_engine = SmartSearchEngine()
            print("Search Engine loaded.")
        except Exception as e:
            print(f"Failed to load Search Engine: {e}")
    return search_engine

def get_body_estimator():
    global body_estimator
    if body_estimator is None:
        try:
            print("Lazy Loading Body Estimator...")
            from ai_modules.body_measurement.estimator import BodyMeasurementEstimator
            body_estimator = BodyMeasurementEstimator()
            print("Body Estimator loaded.")
        except Exception as e:
            print(f"Failed to load Body Estimator: {e}")
    return body_estimator

def get_tryon_engine():
    global tryon_engine
    if tryon_engine is None:
        try:
            print("Lazy Loading Try-On Engine...")
            from ai_modules.virtual_try_on.engine import VirtualTryOnEngine
            tryon_engine = VirtualTryOnEngine()
            print("Try-On Engine loaded.")
        except Exception as e:
            print(f"Failed to load Try-On Engine: {e}")
    return tryon_engine

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up... (Force Redeploy)")
    
    # 1. Load Lightweight Data (Instant)
    load_data()
    
    # NOTE: NO HEAVY MODEL LOADING HERE!
    # Models will be loaded on-demand (lazy) inside endpoints.
            
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
        
        # Log success to recent_logs
        log_entry = {
            "method": request.method,
            "path": endpoint,
            "status": response.status_code,
            "latency": f"{process_time*1000:.1f}ms",
            "timestamp": time.time()
        }
        metrics["recent_logs"].insert(0, log_entry) # Insert at beginning
        if len(metrics["recent_logs"]) > 50:
            metrics["recent_logs"] = metrics["recent_logs"][:50]

        return response
    except Exception as e:
        process_time = time.time() - start_time
        metrics["errors"] += 1
        
        # Log error to recent_logs
        log_entry = {
            "method": request.method,
            "path": endpoint,
            "status": 500,
            "latency": f"{process_time*1000:.1f}ms",
            "timestamp": time.time(),
            "error": str(e)
        }
        metrics["recent_logs"].insert(0, log_entry)
        if len(metrics["recent_logs"]) > 50:
             metrics["recent_logs"] = metrics["recent_logs"][:50]
             
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
    # Health check should be fast. Don't trigger loading here unless necessary.
    return {"status": "ok", "services": {
        "search": search_engine is not None,
        "body_measurement": body_estimator is not None,
        "virtual_try_on": tryon_engine is not None,
        "products_loaded": len(product_ids)
    }}

@app.get("/products")
async def get_products(
    limit: int = 50, 
    gender: Optional[str] = None, 
    category: Optional[str] = None,
    subCategory: Optional[str] = None,
    masterCategory: Optional[str] = None
):
    db = SessionLocal()
    try:
        query = db.query(Product)
        
        if gender:
            query = query.filter(func.lower(Product.gender) == gender.lower())
        
        if category:
            # Check both category and articleType columns
            query = query.filter(or_(
                func.lower(Product.category) == category.lower(),
                func.lower(Product.articleType) == category.lower()
            ))
            
        if subCategory:
            query = query.filter(func.lower(Product.subCategory) == subCategory.lower())

        if masterCategory:
            query = query.filter(func.lower(Product.masterCategory) == masterCategory.lower())
        
        # Efficient Random Sort using DB random() or ID desc
        if limit <= 100:
             query = query.order_by(func.random())
        else:
             query = query.order_by(Product.id)
        
        results = query.limit(limit).all()
        
        # Convert to dict
        final_products = [p.to_dict() for p in results]
        
        return {"results": final_products, "total": len(final_products)}
        
    except Exception as e:
        print(f"Error fetching products: {e}")
        return {"results": [], "total": 0}
    finally:
        db.close()

# Helper: Diversify Results
def diversify_results(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Reranks results to ensure category diversity.
    Prioritizes showing a mix of subCategories/masterCategories.
    """
    if not results:
        return []
        
    buckets = {}
    
    # Group by subCategory (or masterCategory if subCategory is missing)
    for p in results:
        cat = p.get("subCategory") or p.get("masterCategory", "Others")
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(p)
        
    final_list = []
    keys = list(buckets.keys())
    
    # Round-robin selection
    while len(final_list) < limit and any(buckets.values()):
        for k in keys:
            if buckets[k]:
                final_list.append(buckets[k].pop(0))
                if len(final_list) >= limit:
                    break
                    
    return final_list

@app.get("/search")
async def search(
    q: str = Query(...), 
    limit: int = Query(20),
    use_ai: bool = Query(True)
):
    """
    Text search for products.
    """
    results = []
    
    # 1. AI Search (CLIP)
    if use_ai:
        engine = get_search_engine()
        # Check if engine loaded AND we have embeddings AND we have IDs to map to
        if engine and engine.model is not None and product_embeddings is not None and product_ids:
            try:
                # Fetch more candidates to allow for diversity reranking (3x limit)
                search_limit = limit * 3
                
                # search_engine.search returns (values, indices)
                search_res = engine.search(q, product_embeddings, top_k=search_limit)
                indices = search_res[1]
                if hasattr(indices, 'cpu'):
                    indices = indices.cpu().numpy()
                
                # Map Indices -> DB IDs
                found_db_ids = []
                for idx in indices:
                    idx_val = int(idx)
                    if idx_val < len(product_ids):
                        found_db_ids.append(product_ids[idx_val])
                
                # Fetch full objects from DB
                if found_db_ids:
                    db = SessionLocal()
                    try:
                        # Fetch in one query
                        db_results = db.query(Product).filter(Product.id.in_(found_db_ids)).all()
                        db_map = {p.id: p.to_dict() for p in db_results}
                        
                        candidates = []
                        for pid in found_db_ids:
                            if pid in db_map:
                                candidates.append(db_map[pid])
                                
                        # Apply Diversity Reranking
                        results = diversify_results(candidates, limit)
                        return {"query": q, "results": results, "method": "ai_diversified"}
                    finally:
                        db.close()
                        
            except Exception as e:
                print(f"AI Search failed: {e}")
                # Fallback to keyword
            
    # 2. Keyword Fallback (Database ILIKE)
    db = SessionLocal()
    try:
        q_lower = f"%{q}%"
        query = db.query(Product).filter(or_(
            Product.name.ilike(q_lower),
            Product.articleType.ilike(q_lower),
            Product.category.ilike(q_lower)
        ))
        
        # Limit result for diversity processing
        keyword_candidates = query.limit(limit * 3).all()
        candidates_dicts = [p.to_dict() for p in keyword_candidates]
        
        # Apply Diversity Reranking
        results = diversify_results(candidates_dicts, limit)
                    
        return {"query": q, "results": results, "method": "keyword_diversified"}
    except Exception as e:
        print(f"Keyword search failed: {e}")
        return {"query": q, "results": [], "method": "error"}
    finally:
        db.close()

@app.get("/search/suggestions")
def get_suggestions(q: str, limit: int = 5):
    db = SessionLocal()
    try:
        q_lower = f"%{q}%"
        results = db.query(Product.name).filter(Product.name.ilike(q_lower)).limit(limit).all()
        suggestions = [r[0] for r in results]
        return {"query": q, "suggestions": suggestions}
    except Exception as e:
        return {"query": q, "suggestions": []}
    finally:
        db.close()

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), limit: int = 20):
    """
    Semantic search using an image (Image-to-Text).
    """
    engine = get_search_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform search
        # Pass product_histograms if available (global variable)
        ph = product_histograms if 'product_histograms' in globals() else None
        results = engine.search(image, product_embeddings, product_histograms=ph, top_k=limit)
        
        found_db_ids = []
        scores = {}
        
        if engine.model is None:
            # Histogram search returns list of (pid_index, score)
            # Standard implementation usually returns indices into the array passed.
            for idx, score in results:
                idx_val = int(idx)
                if idx_val < len(product_ids):
                    pid = product_ids[idx_val]
                    found_db_ids.append(pid)
                    scores[pid] = float(score)
        else:
            # Embedding search returns (values, indices)
            for idx in results[1]:
                idx_val = int(idx)
                if idx_val < len(product_ids):
                    found_db_ids.append(product_ids[idx_val])
        
        # Fetch from DB
        search_results = []
        if found_db_ids:
            db = SessionLocal()
            try:
                db_results = db.query(Product).filter(Product.id.in_(found_db_ids)).all()
                db_map = {p.id: p.to_dict() for p in db_results}
                
                for pid in found_db_ids:
                    if pid in db_map:
                        item = db_map[pid]
                        if pid in scores:
                            item["score"] = scores[pid]
                        search_results.append(item)
            finally:
                db.close()
            
        return {"query": "image_upload", "results": search_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/measure")
async def measure_body(
    file: UploadFile = File(...),
    height: float = Form(170.0) # Default to 170cm if not provided
):
    """
    Estimate body measurements from an uploaded image.
    """
    estimator = get_body_estimator()
    if not estimator:
        raise HTTPException(status_code=503, detail="Body measurement estimator not initialized")
    
    try:
        import cv2
        # Read image
        contents = await file.read()
        
        # Convert to numpy array (BGR for OpenCV/YOLO)
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_np is None:
             raise HTTPException(status_code=400, detail="Invalid image data")

        # Perform estimation
        # Pass user provided height to estimator
        results = estimator.estimate_from_image(image_np, user_height_cm=height)
        
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
    try:
        print(f"Recommend Request for Product ID: {product_id}")
        
        # Find product
        source_product = next((p for p in products if p["id"] == product_id), None)
                
        if not source_product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        # --- 1. Find Similar Items ---
        similar_items = []
        
        # Try Semantic Similarity first (if available)
        engine = get_search_engine()
        if engine and engine.model is not None and product_embeddings is not None:
            try:
                # Find index of source product
                # Assuming product_embeddings aligns exactly with products list
                # This assumption might be risky if lists drift, but okay for now if static
                source_idx = next((i for i, p in enumerate(products) if p["id"] == product_id), None)
                
                if source_idx is not None:
                    source_embedding = product_embeddings[source_idx]
                    # Search
                    results = engine.search_by_embedding(source_embedding, product_embeddings, top_k=limit+1)
                    
                    # Extract products (excluding self)
                    # results is (values, indices)
                    # Ensure we handle torch tensor or numpy array output from search_by_embedding
                    indices = results[1]
                    if hasattr(indices, 'cpu'):
                        indices = indices.cpu().numpy()
                    elif hasattr(indices, 'numpy'):
                        indices = indices.numpy()
                    
                    for idx in indices:
                        # Handle scalar numpy types or tensors
                        idx_val = int(idx.item()) if hasattr(idx, 'item') else int(idx)
                        
                        if idx_val != source_idx and idx_val < len(products):
                            similar_items.append(products[idx_val])
                    
                    # Limit
                    similar_items = similar_items[:limit]
                    print(f"Found {len(similar_items)} similar items via AI.")
            except Exception as e:
                print(f"Error in semantic recommendation: {e}")
                similar_items = []

        # Fallback to Metadata Similarity if Semantic failed or returned nothing
        if not similar_items:
            print("Fallback to metadata similarity...")
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
        
        # Enforce strict complementary rules
        if source_sub == "Topwear":
            target_subs = ["Bottomwear", "Shoes", "Watches", "Belts", "Eyewear"]
        elif source_sub == "Bottomwear":
            target_subs = ["Topwear", "Shoes", "Watches", "Belts"]
        elif source_sub == "Shoes":
            target_subs = ["Topwear", "Bottomwear", "Watches"]
        elif source_sub == "Dress":
            target_subs = ["Shoes", "Watches", "Jewellery", "Bags"]
        elif source_sub == "Saree":
            target_subs = ["Jewellery", "Bags", "Shoes"]
        else:
            # Fallback for accessories etc. -> match with Apparel
            target_subs = ["Topwear", "Bottomwear", "Dress"]
            
        scored_matches = []
        
        # Neutral colors that match with everything
        neutrals = ["Black", "White", "Grey", "Navy Blue", "Beige", "Silver", "Gold"]
        
        for p in candidates:
            # Strictly filter by target subcategories
            # Allow Accessories and Footwear as generic fallback master categories if subCategory is niche
            if p.get("subCategory") not in target_subs and p.get("masterCategory") not in ["Accessories", "Footwear"]:
                 if p.get("subCategory") not in target_subs:
                    continue
                
            score = 0
            
            # Boost complementary category diversity
            # (e.g. if we have a Top, a Bottom is worth more than a Watch)
            if source_sub == "Topwear" and p.get("subCategory") == "Bottomwear":
                score += 25
            elif source_sub == "Bottomwear" and p.get("subCategory") == "Topwear":
                score += 25
            
            # Context Match
            if p.get("usage") == source_product.get("usage"):
                score += 10
                
            # Strict Season Matching
            # Only allow matching if seasons are compatible or one is 'All'/'Summer'/'Winter' compatible
            s_season = source_product.get("season")
            p_season = p.get("season")
            
            if s_season == p_season:
                 score += 15 # Boost same season
            elif s_season == "Summer" and p_season == "Winter":
                 score -= 100 # Penalize Summer + Winter mismatch
            elif s_season == "Winter" and p_season == "Summer":
                 score -= 100 # Penalize Winter + Summer mismatch
            
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
        
        # Try to pick a mix of categories if possible (e.g. 1 Bottom, 1 Shoe, 1 Accessory)
        matching_items = []
        seen_subs = set()
        
        # First pass: try to get distinct subcategories (Prioritize Outfit Construction)
        for score, p in scored_matches:
            sub = p.get("subCategory")
            if sub not in seen_subs:
                matching_items.append(p)
                seen_subs.add(sub)
                
        # Fill up the rest with high scoring items if needed
        for score, p in scored_matches:
            if len(matching_items) >= limit:
                break
            if p not in matching_items:
                matching_items.append(p)
                
        matching_items = matching_items[:limit]
                    
        return {
            "source_product_id": product_id, 
            "recommendations": similar_items,
            "matching_items": matching_items
        }
    except Exception as e:
        print(f"❌ Critical Error in recommend_products: {e}")
        # Return empty list instead of 500 to keep frontend alive
        return {
            "source_product_id": product_id, 
            "recommendations": [],
            "matching_items": []
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
    engine = get_tryon_engine()
    if not engine:
         raise HTTPException(status_code=503, detail="Try-On engine not initialized")
    
    try:
        import cv2
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
        segmented = engine.segment_cloth(image_np, cloth_type="upper")
        
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

    vto_engine = get_tryon_engine()
    bme_estimator = get_body_estimator()

    if not vto_engine or not bme_estimator:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        import cv2
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
            
            if os.path.exists(cloth_path):
                 cloth_np = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)
            else:
                 # Check if we have the URL in DB (refactored from products list)
                 db = SessionLocal()
                 try:
                     product = db.query(Product).filter(Product.id == cloth_id).first()
                     if product and product.image_url:
                         try:
                             print(f"Downloading image for {cloth_id} from {product.image_url}...")
                             import requests
                             headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                             resp = requests.get(product.image_url, headers=headers, timeout=10)
                             if resp.status_code == 200:
                                 # Convert to numpy
                                 image_bytes = np.frombuffer(resp.content, np.uint8)
                                 cloth_np = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
                                 
                                 # Cache it locally
                                 if cloth_np is not None:
                                     cv2.imwrite(cloth_path, cloth_np)
                                     print(f"Cached image to {cloth_path}")
                             else:
                                 print(f"Failed to download image: {resp.status_code}")
                         except Exception as e:
                             print(f"Error downloading image: {e}")
                 finally:
                     db.close()

                 if cloth_np is None:
                     raise HTTPException(status_code=404, detail=f"Product image for ID {cloth_id} not found locally or remotely")
            
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
        pose_result = bme_estimator.estimate_from_image(person_np)
        
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
        
        result_np = vto_engine.try_on(person_np, cloth_np, keypoints, adjustments=adjustments, use_cloud=use_cloud_bool, hf_token=hf_token)
        
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
