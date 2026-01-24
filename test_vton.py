import requests
import cv2
import numpy as np
import os

def test_segmentation():
    url = "http://localhost:8000/segment"
    
    # We need a test image. Let's look for one in static/products or use a dummy one.
    # If we don't have one, we can't test properly.
    # Let's check if there are images in backend/static/products
    static_dir = "backend/static/products"
    test_image_path = None
    
    if os.path.exists(static_dir):
        files = os.listdir(static_dir)
        if files:
            test_image_path = os.path.join(static_dir, files[0])
            print(f"Using test image: {test_image_path}")
    
    if not test_image_path:
        print("No test image found.")
        return

    # Test with cloth_id
    cloth_id = os.path.splitext(os.path.basename(test_image_path))[0]
    print(f"Testing segmentation for cloth_id: {cloth_id}")
    
    try:
        response = requests.post(url, data={"cloth_id": cloth_id})
        if response.status_code == 200:
            print("Segmentation successful (Status 200)")
            # Save result to verify size/content (optional)
            with open("test_segment_result.png", "wb") as f:
                f.write(response.content)
            print("Saved result to test_segment_result.png")
        else:
            print(f"Segmentation failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_segmentation()
