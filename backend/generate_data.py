import json
import random

# Base Images Mapping (using existing 12 images)
# 1. White T-Shirt
# 2. Blue Jeans
# 3. Floral Dress
# 4. Leather Jacket
# 5. Red Sneakers
# 6. Red Evening Gown
# 7. Grey Hoodie
# 8. Navy Suit
# 9. Beige Trench Coat
# 10. Flannel Shirt
# 11. Black Leggings
# 12. Brown Boots

IMAGE_MAP = {
    "T-Shirt": [1],
    "Jeans": [2],
    "Dress": [3, 6],
    "Jacket": [4, 9],
    "Sneakers": [5],
    "Hoodie": [7],
    "Suit": [8],
    "Shirt": [1, 10],
    "Leggings": [11],
    "Boots": [12],
    "Top": [1, 7, 10],
    "Bottom": [2, 11],
    "Shoes": [5, 12]
}

ADJECTIVES = [
    "Casual", "Formal", "Urban", "Vintage", "Modern", "Classic", "Stylish", "Elegant", 
    "Comfortable", "Premium", "Luxury", "Essential", "Basic", "Trendy", "Chic", "Sporty",
    "Athletic", "Cozy", "Lightweight", "Durable", "Sleek", "Minimalist", "Bohemian", "Retro"
]

COLORS = [
    "Red", "Blue", "Black", "White", "Green", "Yellow", "Grey", "Navy", "Beige", "Brown", 
    "Purple", "Pink", "Orange", "Teal", "Maroon", "Charcoal", "Cream", "Olive", "Silver", "Gold"
]

CATEGORIES = [
    ("T-Shirt", "Tops", "A soft and breathable cotton t-shirt."),
    ("Jeans", "Bottoms", "Classic denim jeans with a perfect fit."),
    ("Dress", "Dresses", "A beautiful dress for any occasion."),
    ("Jacket", "Outerwear", "Keep warm and stylish with this jacket."),
    ("Sneakers", "Footwear", "Comfortable sneakers for daily wear."),
    ("Hoodie", "Tops", "A warm pullover hoodie."),
    ("Suit", "Suits", "Sharp and professional suit."),
    ("Shirt", "Tops", "A versatile shirt for work or play."),
    ("Leggings", "Bottoms", "Stretchy and comfortable leggings."),
    ("Boots", "Footwear", "Rugged boots for all terrains.")
]

def get_image_url(product_name):
    # Try to find a matching image based on keywords in the name
    name_lower = product_name.lower()
    
    candidates = []
    
    if "dress" in name_lower or "gown" in name_lower:
        candidates = IMAGE_MAP["Dress"]
    elif "jean" in name_lower or "denim" in name_lower:
        candidates = IMAGE_MAP["Jeans"]
    elif "jacket" in name_lower or "coat" in name_lower:
        candidates = IMAGE_MAP["Jacket"]
    elif "sneaker" in name_lower or "runner" in name_lower:
        candidates = IMAGE_MAP["Sneakers"]
    elif "boot" in name_lower:
        candidates = IMAGE_MAP["Boots"]
    elif "suit" in name_lower:
        candidates = IMAGE_MAP["Suit"]
    elif "legging" in name_lower or "yoga" in name_lower:
        candidates = IMAGE_MAP["Leggings"]
    elif "hoodie" in name_lower:
        candidates = IMAGE_MAP["Hoodie"]
    elif "flannel" in name_lower or "check" in name_lower:
        candidates = [10]
    elif "shirt" in name_lower or "tee" in name_lower:
        candidates = IMAGE_MAP["Shirt"]
    
    if not candidates:
        # Fallback based on generic logic
        candidates = [1] 
        
    img_id = random.choice(candidates)
    return f"http://localhost:8000/static/images/{img_id}.jpg"

products = []
count = 0

print("Generating 5000 products...")

while count < 5000:
    adj = random.choice(ADJECTIVES)
    color = random.choice(COLORS)
    cat_name, cat_group, cat_desc = random.choice(CATEGORIES)
    
    name = f"{adj} {color} {cat_name}"
    
    # Ensure uniqueness in name if possible, or just accept collisions (5000 is high for these lists)
    # Let's add a variant ID or something to make it unique if needed, or just allow similar items.
    # To ensure 5000 list items, we can just append.
    
    # Refine description
    description = f"{cat_desc} This {color.lower()} {cat_name.lower()} features a {adj.lower()} design."
    
    image_url = get_image_url(name)
    
    product = {
        "id": count + 1,
        "name": name,
        "description": description,
        "category": cat_group,
        "image_url": image_url
    }
    
    products.append(product)
    count += 1

output_path = "products.json"
with open(output_path, "w") as f:
    json.dump(products, f, indent=2)

print(f"Successfully generated {len(products)} products in {output_path}")
