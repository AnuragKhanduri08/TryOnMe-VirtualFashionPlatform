import cv2
import numpy as np
from ai_modules.virtual_try_on.engine import VirtualTryOnEngine
import os

def test_local_warp():
    print("Testing Enhanced Local Warping...")
    engine = VirtualTryOnEngine()
    
    # Create dummy person image (white background)
    person_img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    # Draw a dummy body
    # Head
    cv2.circle(person_img, (300, 100), 50, (200, 200, 200), -1)
    # Torso
    cv2.rectangle(person_img, (250, 150), (350, 400), (200, 200, 200), -1)
    
    # Create dummy cloth image (Red T-Shirt)
    cloth_img = np.ones((500, 500, 3), dtype=np.uint8) * 0 # Black background
    # Draw red rectangle as shirt
    cv2.rectangle(cloth_img, (100, 100), (400, 400), (0, 0, 255), -1)
    
    # Dummy Keypoints (YOLO Format: [x, y, conf])
    # 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip
    keypoints = [[0,0,0]] * 17
    keypoints[5] = [250, 150, 0.9] # L-Shoulder
    keypoints[6] = [350, 150, 0.9] # R-Shoulder
    keypoints[11] = [250, 400, 0.9] # L-Hip
    keypoints[12] = [350, 400, 0.9] # R-Hip
    
    # Run Try-On (Local Mode)
    try:
        result = engine.try_on(person_img, cloth_img, keypoints, use_cloud=False)
        print("Local Try-On Successful.")
        cv2.imwrite("test_local_warp_result.jpg", result)
        print("Result saved to test_local_warp_result.jpg")
    except Exception as e:
        print(f"Local Try-On Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_warp()
