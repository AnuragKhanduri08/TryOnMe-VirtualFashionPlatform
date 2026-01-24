from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
print("Downloading Segformer model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
print("Download complete.")
