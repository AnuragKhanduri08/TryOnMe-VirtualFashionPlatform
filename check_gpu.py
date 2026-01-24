import torch
import sys

def check_gpu():
    print("Checking System Capabilities for GAN/Diffusion Models...")
    print(f"Python: {sys.version.split()[0]}")
    
    try:
        is_cuda = torch.cuda.is_available()
        print(f"CUDA Available: {is_cuda}")
        
        if is_cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}")
            print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
            
            if props.total_memory / 1024**3 < 8:
                print("\n[WARNING] VRAM is less than 8GB. VITON-HD/Diffusion models may OOM (Out of Memory).")
            else:
                print("\n[SUCCESS] GPU is powerful enough for local inference.")
        else:
            print("\n[INFO] No GPU detected. Running complex GANs locally will be extremely slow (CPU mode).")
            print("Recommendation: Use Cloud API.")
            
    except Exception as e:
        print(f"Error checking torch: {e}")

if __name__ == "__main__":
    check_gpu()
