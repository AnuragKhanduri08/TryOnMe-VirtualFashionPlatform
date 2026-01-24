from gradio_client import Client
import shutil
import os

def test_catvton():
    print("Testing Alternative Cloud APIs...")
    
    # List of potential spaces to try
    spaces = [
        "Kwai-Kolors/Kolors-Virtual-Try-On",
        "yisol/IDM-VTON",
    ]
    
    for space_id in spaces:
        try:
            print(f"\n--------------------------------")
            print(f"Connecting to {space_id}...")
            client = Client(space_id)
            print(f"Connected successfully to {space_id}!")
            
            # List API endpoints
            client.view_api()
            
        except Exception as e:
            print(f"Failed to connect to {space_id}: {e}")

if __name__ == "__main__":
    test_catvton()
