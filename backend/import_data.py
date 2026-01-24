import os
import json
import csv
import pandas as pd
import numpy as np

def import_dataset():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "static", "products")
    products_json_path = os.path.join(base_dir, "products.json")
    csv_path = os.path.join(base_dir, "dataset.csv") # Default expected name
    
    # 1. Verify Images
    if not os.path.exists(images_dir):
        print("Images directory not found!")
        return
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = {f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_extensions}
    print(f"Found {len(image_files)} images in static/products/")
    
    products = []
    
    # 2. Check for CSV
    if os.path.exists(csv_path):
        print(f"Found CSV file: {csv_path}")
        try:
            # Attempt to read CSV with pandas for robustness
            # Use on_bad_lines='skip' to ignore malformed lines
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            print(f"CSV Columns: {df.columns.tolist()}")
            
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Map columns to our schema
            # Expected schema: id, name, category, description, image (filename)
            
            for index, row in df.iterrows():
                # Heuristic column mapping
                p_id = row.get('id', index + 1)
                
                # Name: prefer 'product_name', 'name', 'title', 'productdisplayname'
                name = row.get('product_name') or row.get('name') or row.get('title') or row.get('productdisplayname') or f"Product {p_id}"
                
                # Category: prefer 'category', 'sub_category', 'articleType', 'masterCategory'
                category = row.get('category') or row.get('sub_category') or row.get('articletype') or row.get('mastercategory') or "Uncategorized"
                
                # Description: prefer 'description', 'product_details', or construct from metadata
                description = row.get('description') or row.get('product_details')
                if not description:
                    # Construct description from available metadata
                    parts = []
                    if row.get('gender'): parts.append(f"For {row.get('gender')}")
                    if row.get('basecolour'): parts.append(f"Color: {row.get('basecolour')}")
                    if row.get('usage'): parts.append(f"Usage: {row.get('usage')}")
                    if row.get('season'): parts.append(f"Season: {row.get('season')}")
                    description = f"{name}. {', '.join(parts)}." if parts else f"A stylish {name}."
                
                # Image: prefer 'image', 'filename', 'id' (if id matches filename)
                image_ref = str(row.get('image') or row.get('filename') or row.get('id') or "")
                
                # Try to find the matching image file
                # 1. Direct match
                if image_ref in image_files:
                    image_filename = image_ref
                # 2. Match with extension added
                elif f"{image_ref}.jpg" in image_files:
                    image_filename = f"{image_ref}.jpg"
                elif f"{image_ref}.png" in image_files:
                    image_filename = f"{image_ref}.png"
                # 3. If image_ref is a full ID (e.g. 1163), check 1163.jpg
                else:
                    # Fallback: if we have IDs in the CSV that match filenames
                    # e.g. ID 1163 -> 1163.jpg
                    potential_name = f"{p_id}.jpg"
                    if potential_name in image_files:
                        image_filename = potential_name
                    else:
                        # If no match found, use a placeholder or skip? 
                        # For now, let's just pick one randomly or leave empty?
                        # Better: Skip image assignment if not found, or use default
                        image_filename = None

                if image_filename:
                    image_url = f"http://localhost:8000/static/products/{image_filename}"
                else:
                    # Fallback to random image if specific one not found (Optional)
                    # or just keep it None and handle in UI
                    image_url = "http://localhost:8000/static/images/1.jpg" # Default fallback
                
                products.append({
                    "id": int(p_id) if str(p_id).isdigit() else index + 1,
                    "name": str(name),
                    "description": str(description),
                    "category": str(category),
                    "image_url": image_url
                })
                
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return
    else:
        print("CSV file 'dataset.csv' not found in backend folder.")
        print("Falling back to generating products from image filenames...")
        
        sorted_images = sorted(list(image_files))
        
        for idx, img_name in enumerate(sorted_images):
            # Infer details from filename
            # e.g. "Red_Dress.jpg" -> Name: "Red Dress"
            name_part = os.path.splitext(img_name)[0]
            clean_name = name_part.replace("_", " ").replace("-", " ").title()
            
            # Simple category heuristic based on name keywords
            category = "General"
            name_lower = clean_name.lower()
            if "dress" in name_lower: category = "Dresses"
            elif "shirt" in name_lower or "top" in name_lower: category = "Tops"
            elif "pant" in name_lower or "jeans" in name_lower: category = "Bottoms"
            elif "shoe" in name_lower or "boot" in name_lower: category = "Footwear"
            elif "jacket" in name_lower or "coat" in name_lower: category = "Outerwear"
            elif "bag" in name_lower: category = "Accessories"
            elif "watch" in name_lower: category = "Accessories"
            
            products.append({
                "id": idx + 1,
                "name": clean_name,
                "description": f"A high-quality {clean_name}.",
                "category": category,
                "image_url": f"http://localhost:8000/static/products/{img_name}"
            })

    # 3. Save to products.json
    print(f"Saving {len(products)} products to products.json...")
    with open(products_json_path, "w") as f:
        json.dump(products, f, indent=4)
        
    print("Done! Please restart the backend to load the new products.")

if __name__ == "__main__":
    import_dataset()
