import sys
import os
import cv2
import numpy as np
import torch

# Add ai_modules to path
sys.path.append(os.path.join(os.getcwd(), "ai_modules"))

from virtual_try_on.engine import VirtualTryOnEngine

from unittest.mock import patch

def test_segmentation():
    print("Initializing Engine (Mocking Cloud)...")
    
    # Patch the connect_to_cloud method to avoid hanging/exiting
    with patch('virtual_try_on.engine.VirtualTryOnEngine.connect_to_cloud') as mock_connect:
        engine = VirtualTryOnEngine()
        print("Engine initialized.")
        
        print(f"Current working directory: {os.getcwd()}")
    img_path = os.path.abspath("backend/static/products/15970.jpg")
    print(f"Checking image at: {img_path}")
    if not os.path.exists(img_path):
        print(f"Image NOT found at {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image (cv2 returned None)")
        return
        
    print(f"Testing segmentation on {img_path} (Shape: {img.shape})")
    
    try:
        # Test Upper Cloth Segmentation
        segmented = engine.segment_cloth(img, cloth_type="upper")
    except Exception as e:
        print(f"Segmentation error: {e}")
        return
    
    if segmented is not None:
        output_path = "test_segmented_upper.png"
        cv2.imwrite(output_path, segmented)
        print(f"Saved {output_path}")
        
        # Verify alpha channel
        if segmented.shape[2] == 4:
            alpha = segmented[:, :, 3]
            non_zero = cv2.countNonZero(alpha)
            print(f"Non-zero alpha pixels: {non_zero}")
            if non_zero == 0:
                print("WARNING: Alpha channel is completely transparent!")
        else:
            print("Output is not BGRA!")
            
    else:
        print("Segmentation failed (returned None)")

if __name__ == "__main__":
    test_segmentation()
