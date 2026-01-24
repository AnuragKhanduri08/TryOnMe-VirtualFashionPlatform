
from gradio_client import Client, handle_file
import time
import cv2
import numpy as np

def test_outfit_anyone():
    print("Testing HumanAIGC/OutfitAnyone...")
    client = Client("HumanAIGC/OutfitAnyone")
    
    # Create dummy images
    person = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.imwrite("dummy_p.jpg", person)
    cloth = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.imwrite("dummy_c.jpg", cloth)
    
    try:
        # predict(model_name, garment1, garment2, api_name="/get_tryon_result")
        # Note: OutfitAnyone takes model_name (image), garment1 (top), garment2 (bottom)
        # We only have one cloth. We might need to pass it as both or just one.
        
        print("Sending request...")
        res = client.predict(
            handle_file("dummy_p.jpg"), # model
            handle_file("dummy_c.jpg"), # top
            handle_file("dummy_c.jpg"), # bottom
            api_name="/get_tryon_result"
        )
        print(f"Success: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_outfit_anyone()
