
import cv2
import numpy as np
import tempfile
import os
import time
from gradio_client import Client, handle_file

def create_dummy_images():
    # Create dummy person image (White background, Black stick figure)
    person = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.circle(person, (192, 100), 40, (0, 0, 0), -1) # Head
    cv2.line(person, (192, 140), (192, 300), (0, 0, 0), 20) # Body
    
    # Create dummy cloth image (Red t-shirt shape)
    cloth = np.ones((512, 384, 3), dtype=np.uint8) * 255
    cv2.rectangle(cloth, (100, 100), (284, 300), (0, 0, 255), -1)
    
    cv2.imwrite("dummy_person.jpg", person)
    cv2.imwrite("dummy_cloth.jpg", cloth)
    return "dummy_person.jpg", "dummy_cloth.jpg"

def test_provider(name, client_id, api_fn):
    print(f"\n--- Testing {name} ({client_id}) ---")
    person_path, cloth_path = create_dummy_images()
    
    try:
        start_time = time.time()
        client = Client(client_id)
        print(f"Connected in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        result = api_fn(client, person_path, cloth_path)
        print(f"Prediction took {time.time() - start_time:.2f}s")
        
        if result:
            print("SUCCESS: Got result!")
            return True
        else:
            print("FAILED: No result returned.")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def call_idm_vton(client, p, c):
    # predict(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, api_name="/tryon")
    res = client.predict(
        {"background": handle_file(p), "layers": [], "composite": None},
        handle_file(c),
        "shirt", True, False, 30, 42,
        api_name="/tryon"
    )
    return res

def call_ootd_hd(client, p, c):
    # predict(vton_img, garm_img, n_samples, n_steps, image_scale, seed, api_name="/process_hd")
    return client.predict(
        handle_file(p), handle_file(c), 1, 20, 2.0, -1,
        api_name="/process_hd"
    )

def call_kolors(client, p, c):
    # Based on API view for kwai-Kolors/Kolors-Virtual-Try-On
    # Usually it's /tryon or similar. I need to probe it first or guess.
    # Let's try standard signature or check view_api in logs
    # For now, let's just connect and view_api to confirm signature
    client.view_api()
    return None

if __name__ == "__main__":
    # 1. IDM-VTON (The one we use)
    test_provider("IDM-VTON", "yisol/IDM-VTON", call_idm_vton)
    
    # 2. OOTDiffusion (Fallback)
    test_provider("OOTDiffusion", "levihsu/OOTDiffusion", call_ootd_hd)
    
    # 3. Kolors (Potential New)
    # test_provider("Kolors", "kwai-Kolors/Kolors-Virtual-Try-On", call_kolors)
