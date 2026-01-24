import pandas as pd
import json
import os

def import_kaggle_data():
    # Paths - Use absolute paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset.csv')
    print(f"Base Dir: {base_dir}")
    print(f"Looking for dataset at: {dataset_path}")
    images_csv_path = os.path.join(base_dir, 'images.csv')
    output_json_path = os.path.join(base_dir, 'products.json')

    # Check if dataset.csv exists
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    # Check if images.csv exists
    if not os.path.exists(images_csv_path):
        print(f"Error: {images_csv_path} not found. Please download it from Kaggle.")
        print("Run: kaggle datasets download paramaggarwal/fashion-product-images-dataset -f fashion-dataset/images.csv")
        return

    print("Loading datasets...")
    try:
        styles_df = pd.read_csv(dataset_path, on_bad_lines='skip')
        images_df = pd.read_csv(images_csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    print(f"Styles: {len(styles_df)} rows")
    print(f"Images: {len(images_df)} rows")

    # Clean up column names
    styles_df.columns = styles_df.columns.str.strip()
    images_df.columns = images_df.columns.str.strip()

    # Ensure ID is treated as string for matching if needed, or int
    # styles.csv 'id' is int usually.
    # images.csv 'filename' is usually 'id.jpg'.
    
    # Extract ID from filename in images_df
    if 'filename' in images_df.columns:
        images_df['id'] = images_df['filename'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
    
    # Merge
    # We want to keep all styles, and attach URL if available
    merged_df = pd.merge(styles_df, images_df, on='id', how='left')

    # Prepare products list
    products = []
    
    # We need to map 'link' to 'image_url'
    # Check if 'link' exists
    url_col = 'link' if 'link' in merged_df.columns else None
    
    if not url_col:
        print("Warning: 'link' column not found in images.csv. Cannot set remote URLs.")
        # Try to find a URL column
        for col in merged_df.columns:
            if 'url' in col.lower() or 'link' in col.lower():
                url_col = col
                break
    
    print(f"Using '{url_col}' for image URLs.")

    for index, row in merged_df.iterrows():
        try:
            if index % 1000 == 0:
                print(f"Processing row {index}...")
            product_id = row['id']
            # Use remote URL from Kaggle dataset (cloud)
            if url_col and pd.notna(row[url_col]):
                image_url = row[url_col]
            else:
                # Fallback to local if remote URL is missing
                image_url = f"http://localhost:8000/static/products/{product_id}.jpg"
            
            # Construct meaningful description
            desc = f"{row['productDisplayName']}. For {row['gender']}, Color: {row['baseColour']}, Usage: {row['usage']}, Season: {row['season']}."
            
            product = {
                "id": int(product_id),
                "name": str(row['productDisplayName']),
                "description": desc,
                "category": str(row['articleType']), # or subCategory or masterCategory
                "image_url": image_url,
                "gender": str(row['gender']),
                "masterCategory": str(row['masterCategory']),
                "subCategory": str(row['subCategory']),
                "articleType": str(row['articleType']),
                "baseColour": str(row['baseColour']),
                "season": str(row['season']),
                "usage": str(row['usage'])
            }
            products.append(product)
        except Exception as e:
            if index % 1000 == 0:
                print(f"Error on row {index}: {e}")
            continue

    # Save to products.json
    with open(output_json_path, 'w') as f:
        json.dump(products, f, indent=4)
    
    print(f"Successfully exported {len(products)} products to {output_json_path}")

if __name__ == "__main__":
    import_kaggle_data()
