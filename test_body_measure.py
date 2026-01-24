import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.body_measurement.estimator import BodyMeasurementEstimator

def test_measurement():
    print("Testing Body Measurement Module...")
    
    estimator = BodyMeasurementEstimator()
    
    # Create a dummy image with a person-like shape (simplified)
    # White background
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    
    # Draw a stick figure (very rough) to see if mediapipe detects anything
    # Head
    cv2.circle(img, (200, 100), 30, (0, 0, 0), -1)
    # Body
    cv2.line(img, (200, 130), (200, 300), (0, 0, 0), 10)
    # Arms
    cv2.line(img, (150, 150), (250, 150), (0, 0, 0), 10)
    # Legs
    cv2.line(img, (200, 300), (150, 500), (0, 0, 0), 10)
    cv2.line(img, (200, 300), (250, 500), (0, 0, 0), 10)
    
    # Save dummy image
    dummy_path = "test_body.jpg"
    cv2.imwrite(dummy_path, img)
    
    try:
        results = estimator.estimate_from_image(dummy_path)
        print("Results:", results)
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    if os.path.exists(dummy_path):
        os.remove(dummy_path)

if __name__ == "__main__":
    test_measurement()
