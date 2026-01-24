import cv2
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from ai_modules.virtual_try_on.engine import VirtualTryOnEngine

def test_cloud_integration(token=None):
    print("Testing Cloud Try-On Integration...")
    
    # Create dummy person image (White background, Black stick figure)
    person = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.circle(person, (192, 100), 40, (0, 0, 0), -1) # Head
    cv2.line(person, (192, 140), (192, 300), (0, 0, 0), 20) # Body
    
    # Create dummy cloth image (Red t-shirt shape)
    cloth = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.rectangle(cloth, (100, 100), (284, 300), (0, 0, 255), -1)
    
    engine = VirtualTryOnEngine()
    
    print("Sending request to Cloud (this might take 30-60s)...")
    try:
        # Pass the token if provided
        result = engine.try_on(person, cloth, keypoints=[], use_cloud=True, hf_token=token)
        
        if result is not None:
            print("Success! Received result from Cloud.")
            print(f"Result Shape: {result.shape}")
            cv2.imwrite("test_cloud_result.jpg", result)
            print("Saved to test_cloud_result.jpg")
        else:
            print("Failed: Result was None (Check logs/quotas).")
            
    except Exception as e:
        print(f"Test Failed with Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cloud Virtual Try-On")
    parser.add_argument("--token", type=str, help="Hugging Face Token (optional)", default=None)
    args = parser.parse_args()
    
    test_cloud_integration(args.token)
