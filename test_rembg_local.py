import cv2
import numpy as np
import os
import sys

# Add path
sys.path.append(os.path.join(os.getcwd(), "backend"))
sys.path.append(os.path.join(os.getcwd(), "ai_modules"))

try:
    from rembg import remove
    print("rembg imported successfully")
except ImportError:
    print("rembg not installed")
    sys.exit(1)

def test_segmentation(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Processing {image_path}...")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return

    # Convert to RGB (PIL needs RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to bytes
    success, encoded = cv2.imencode(".jpg", img)
    input_bytes = encoded.tobytes()
    
    # Process
    try:
        # rembg expects bytes or PIL image
        output_bytes = remove(input_bytes)
        
        # Convert back to numpy
        nparr = np.frombuffer(output_bytes, np.uint8)
        output_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # Save
        cv2.imwrite(output_path, output_img)
        print(f"Saved result to {output_path}")
        
        # Check alpha
        if output_img.shape[2] == 4:
            print("Output has alpha channel.")
            # Check corners
            corner_alpha = output_img[0, 0, 3]
            print(f"Top-left pixel alpha: {corner_alpha}")
        else:
            print("Output has NO alpha channel.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with a product image
    product_dir = os.path.join("backend", "static", "products")
    # Pick a few files
    files = os.listdir(product_dir)[:3]
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_segmentation(os.path.join(product_dir, f), f"test_seg_{f}.png")
