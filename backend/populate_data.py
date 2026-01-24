import os
import requests
import json

# Enhanced Product Catalog
PRODUCTS = [
    {
        "id": 1,
        "name": "Classic White T-Shirt",
        "description": "A simple, comfortable cotton white t-shirt suitable for casual wear.",
        "category": "Tops",
        "image_url_source": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 2,
        "name": "Blue Denim Jeans",
        "description": "Slim fit blue denim jeans with a classic 5-pocket design.",
        "category": "Bottoms",
        "image_url_source": "https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 3,
        "name": "Floral Summer Dress",
        "description": "A light and airy floral dress perfect for summer outings.",
        "category": "Dresses",
        "image_url_source": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 4,
        "name": "Black Leather Jacket",
        "description": "A stylish black leather jacket for a cool, edgy look.",
        "category": "Outerwear",
        "image_url_source": "https://images.unsplash.com/photo-1551028919-ac66e624ec12?w=500&auto=format&fit=crop&q=60" 
    },
    {
        "id": 5,
        "name": "Red Running Sneakers",
        "description": "Lightweight red sneakers designed for running and sports.",
        "category": "Footwear",
        "image_url_source": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 6,
        "name": "Elegant Red Evening Gown",
        "description": "A stunning red evening gown for formal occasions.",
        "category": "Dresses",
        "image_url_source": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 7,
        "name": "Grey Pullover Hoodie",
        "description": "A cozy grey hoodie made from soft fleece fabric.",
        "category": "Tops",
        "image_url_source": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 8,
        "name": "Navy Blue Business Suit",
        "description": "A sharp navy blue suit for professional settings.",
        "category": "Suits",
        "image_url_source": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 9,
        "name": "Beige Trench Coat",
        "description": "A classic beige trench coat for rainy days and autumn fashion.",
        "category": "Outerwear",
        "image_url_source": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 10,
        "name": "Checked Flannel Shirt",
        "description": "A comfortable red and black checked flannel shirt.",
        "category": "Tops",
        "image_url_source": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 11,
        "name": "Black Yoga Leggings",
        "description": "High-waisted black leggings suitable for yoga and workouts.",
        "category": "Bottoms",
        "image_url_source": "https://images.unsplash.com/photo-1506619216599-9d16d0903dfd?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": 12,
        "name": "Brown Leather Boots",
        "description": "Durable brown leather boots for hiking or casual wear.",
        "category": "Footwear",
        "image_url_source": "https://images.unsplash.com/photo-1542280756-74b2f55e73ab?w=500&auto=format&fit=crop&q=60"
    }
]

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

def populate_data():
    print(f"Downloading images to {STATIC_DIR}...")
    
    final_products = []

    for product in PRODUCTS:
        pid = product["id"]
        url = product["image_url_source"]
        filename = f"{pid}.jpg"
        filepath = os.path.join(STATIC_DIR, filename)
        
        # Download if not exists
        if not os.path.exists(filepath):
            try:
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
        else:
            print(f"Image {filename} already exists.")

        # Update product object with local URL
        product_entry = product.copy()
        del product_entry["image_url_source"]
        product_entry["image_url"] = f"http://localhost:8000/static/images/{filename}"
        final_products.append(product_entry)

    # Write products.json
    json_path = os.path.join(os.path.dirname(__file__), "products.json")
    with open(json_path, "w") as f:
        json.dump(final_products, f, indent=2)
    
    print(f"Successfully wrote {len(final_products)} products to products.json")

if __name__ == "__main__":
    populate_data()
