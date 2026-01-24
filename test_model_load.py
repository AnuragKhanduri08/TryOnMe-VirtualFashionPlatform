
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing torch...", flush=True)
import torch
print("torch imported", flush=True)

print("Importing cv2...", flush=True)
import cv2
print("cv2 imported", flush=True)

from transformers import CLIPModel, CLIPProcessor
print("Imports done", flush=True)

print("Loading processor...", flush=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Processor loaded.", flush=True)

print("Loading model...", flush=True)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Model loaded.", flush=True)
