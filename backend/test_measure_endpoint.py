import requests
from PIL import Image
import io
import cv2
import numpy as np
import os

def test_measure_endpoint():
    print("Testing /measure endpoint...")
    
    # Create a dummy image (white background)
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    # Draw a simple stick figure
    # Head
    cv2.circle(img, (200, 100), 30, (0, 0, 0), -1)
    # Body
    cv2.line(img, (200, 130), (200, 300), (0, 0, 0), 10)
    # Arms
    cv2.line(img, (150, 150), (250, 150), (0, 0, 0), 10)
    # Legs
    cv2.line(img, (200, 300), (150, 500), (0, 0, 0), 10)
    cv2.line(img, (200, 300), (250, 500), (0, 0, 0), 10)

    # Encode to PNG
    success, encoded_image = cv2.imencode('.png', img)
    if not success:
        print("Failed to encode image")
        return

    img_bytes = encoded_image.tobytes()

    url = "http://127.0.0.1:8000/measure"
    files = {'file': ('test_body.png', img_bytes, 'image/png')}
    
    try:
        print("Sending request...")
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_measure_endpoint()
