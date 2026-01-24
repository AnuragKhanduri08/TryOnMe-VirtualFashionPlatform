
import sys
import os
import time

print("Start imports...")
start = time.time()

print("Importing numpy...")
import numpy
print(f"Numpy imported in {time.time()-start:.2f}s")

print("Importing torch...")
import torch
print(f"Torch imported in {time.time()-start:.2f}s")

print("Importing sentence_transformers...")
from sentence_transformers import SentenceTransformer
print(f"SentenceTransformer imported in {time.time()-start:.2f}s")

print("Importing PIL...")
from PIL import Image
print(f"PIL imported in {time.time()-start:.2f}s")

print("Importing cv2...")
import cv2
print(f"CV2 imported in {time.time()-start:.2f}s")

print("All imports successful.")
