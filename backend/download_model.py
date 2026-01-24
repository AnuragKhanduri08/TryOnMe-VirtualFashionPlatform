from sentence_transformers import SentenceTransformer

print("Downloading/Loading model...")
try:
    model = SentenceTransformer('clip-ViT-B-32')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
