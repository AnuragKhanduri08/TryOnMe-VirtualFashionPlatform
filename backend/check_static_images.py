import os
from PIL import Image

def check_static_images():
    base_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images = [f for f in os.listdir(base_dir) if f.endswith(('.jpg', '.png'))][:5]
    
    print(f"Checking sample images from {base_dir}...")
    
    for img_name in images:
        img_path = os.path.join(base_dir, img_name)
        try:
            with Image.open(img_path) as img:
                print(f"Image: {img_name}, Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            print(f"Error reading {img_name}: {e}")

if __name__ == "__main__":
    check_static_images()
