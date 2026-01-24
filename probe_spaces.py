
from gradio_client import Client

def probe_space(space_name):
    print(f"--- Probing {space_name} ---")
    try:
        client = Client(space_name)
        client.view_api()
    except Exception as e:
        print(f"Error connecting to {space_name}: {e}")

if __name__ == "__main__":
    probe_space("levihsu/OOTDiffusion")
    probe_space("HumanAIGC/OutfitAnyone")
    probe_space("yisol/IDM-VTON")
