from gradio_client import Client

def inspect_api():
    try:
        client = Client("yisol/IDM-VTON")
        print("Connected to IDM-VTON Space.")
        client.view_api()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_api()
