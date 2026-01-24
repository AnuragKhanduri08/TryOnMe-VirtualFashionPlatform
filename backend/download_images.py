import os
import requests
import json

# Product Image Mapping (Unsplash URLs)
IMAGE_URLS = {
    1: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=500&auto=format&fit=crop&q=60", # White T-Shirt
    2: "https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=500&auto=format&fit=crop&q=60", # Denim Jeans
    3: "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=500&auto=format&fit=crop&q=60", # Floral Dress
    4: "https://images.unsplash.com/photo-1520975661595-6453be3f7070?w=500&auto=format&fit=crop&q=60", # Leather Jacket
    5: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500&auto=format&fit=crop&q=60", # Sneakers
    6: "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500&auto=format&fit=crop&q=60", # Red Dress
    7: "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=500&auto=format&fit=crop&q=60", # Grey Hoodie
    8: "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500&auto=format&fit=crop&q=60"  # Suit
}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

def download_images():
    print(f"Downloading images to {STATIC_DIR}...")
    
    for product_id, url in IMAGE_URLS.items():
        try:
            filename = f"{product_id}.jpg"
            filepath = os.path.join(STATIC_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"Image {filename} already exists. Skipping.")
                continue
                
            print(f"Downloading {filename}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filename}")
            else:
                print(f"Failed to download {filename}: Status {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

def update_products_json():
    json_path = os.path.join(os.path.dirname(__file__), "products.json")
    if not os.path.exists(json_path):
        print("products.json not found!")
        return

    with open(json_path, "r") as f:
        products = json.load(f)

    updated = False
    for product in products:
        pid = product["id"]
        # Use localhost URL
        local_url = f"http://localhost:8000/static/images/{pid}.jpg"
        if product.get("image_url") != local_url:
            product["image_url"] = local_url
            updated = True
            
    if updated:
        with open(json_path, "w") as f:
            json.dump(products, f, indent=2)
        print("Updated products.json with local image URLs.")
    else:
        print("products.json already up to date.")

if __name__ == "__main__":
    download_images()
    update_products_json()
