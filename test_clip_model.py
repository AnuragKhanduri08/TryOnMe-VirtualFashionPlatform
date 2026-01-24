
import sys
print("Start", flush=True)
import torch
print("Torch imported", flush=True)
from transformers import CLIPConfig
print("CLIPConfig imported", flush=True)
try:
    from transformers import BertModel
    print("BertModel imported", flush=True)
except:
    print("BertModel failed")
from transformers import CLIPModel
print("CLIPModel imported", flush=True)
