import os
from transformers import CLIPProcessor, CLIPModel

def download_models():
    print("========================================")
    print("   PRE-DOWNLOADING AI MODELS")
    print("========================================")
    
    model_name = "openai/clip-vit-base-patch32"
    print(f"Downloading {model_name}...")
    
    # This will download the model to the cache directory defined by HF_HOME
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    print(f"✅ Successfully downloaded and cached {model_name}")

if __name__ == "__main__":
    download_models()
