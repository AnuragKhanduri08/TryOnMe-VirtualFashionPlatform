
from gradio_client import Client, handle_file
import os
import time

def test_connection(space_id):
    print(f"Testing connection to {space_id}...")
    try:
        client = Client(space_id)
        print(f"Successfully connected to {space_id}")
        return client
    except Exception as e:
        print(f"Failed to connect to {space_id}: {e}")
        return None

def test_catvton():
    spaces = ["zhengchong/CatVTON", "SaadAhmedSiddiqui/CatVTON"]
    for s in spaces:
        client = test_connection(s)
        if client:
            print(f"Attempting to view API for {s}...")
            try:
                client.view_api()
            except Exception as e:
                print(f"Error viewing API: {e}")

def test_idm_vton():
    client = test_connection("yisol/IDM-VTON")
    if not client:
        return
    
    print("Attempting to view API for IDM-VTON...")
    try:
        client.view_api()
    except Exception as e:
        print(f"Error viewing API: {e}")

def test_ootd():
    client = test_connection("levihsu/OOTDiffusion")
    if not client:
        return
    
    print("Attempting to view API for OOTDiffusion...")
    try:
        client.view_api()
    except Exception as e:
        print(f"Error viewing API: {e}")

if __name__ == "__main__":
    print("--- Testing CatVTON ---")
    test_catvton()
    print("\n--- Testing IDM-VTON ---")
    test_idm_vton()
    print("\n--- Testing OOTDiffusion ---")
    test_ootd()
