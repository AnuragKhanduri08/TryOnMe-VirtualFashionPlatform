import json
import os
import random

def map_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "static", "products")
    products_path = os.path.join(base_dir, "products.json")
    
    # 1. Get all images
    if not os.path.exists(images_dir):
        print(f"Error: Directory {images_dir} does not exist.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not images:
        print(f"No images found in {images_dir}. Please add images first.")
        return

    print(f"Found {len(images)} images.")

    # 2. Load Products
    if not os.path.exists(products_path):
        print("Error: products.json not found.")
        return

    with open(products_path, "r") as f:
        products = json.load(f)

    print(f"Processing {len(products)} products...")

    # 3. Map Images
    # Strategy: 
    # - If image count >= product count: assign unique image to each
    # - If image count < product count: loop images
    # - Bonus: Try to match filename to category (simple heuristic)
    
    # Sort images to ensure deterministic order if needed, or shuffle for randomness
    random.shuffle(images)
    
    image_pool = images.copy()
    
    updated_count = 0
    for p in products:
        if not image_pool:
            image_pool = images.copy() # Refill pool if exhausted
            random.shuffle(image_pool)
            
        # Pop an image
        img_name = image_pool.pop()
        
        # Construct URL
        # Assuming backend is running on localhost:8000
        p["image_url"] = f"http://localhost:8000/static/products/{img_name}"
        updated_count += 1

    # 4. Save
    with open(products_path, "w") as f:
        json.dump(products, f, indent=4)

    print(f"Successfully updated {updated_count} products with images from static/products/")

if __name__ == "__main__":
    map_images()
