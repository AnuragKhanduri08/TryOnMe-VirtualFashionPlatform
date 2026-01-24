from gradio_client import Client

def inspect_kolors():
    print("Inspecting Kwai-Kolors API...")
    try:
        client = Client("Kwai-Kolors/Kolors-Virtual-Try-On")
        print("Connected.")
        
        # Try to view API with all details
        print(client.view_api(return_format="dict"))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_kolors()
