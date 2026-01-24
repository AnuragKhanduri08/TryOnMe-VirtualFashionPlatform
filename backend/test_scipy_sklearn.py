import sys
print("Importing scipy...", flush=True)
try:
    import scipy
    print(f"Scipy version: {scipy.__version__}", flush=True)
except Exception as e:
    print(f"Error scipy: {e}", flush=True)

print("Importing sklearn...", flush=True)
try:
    import sklearn
    print(f"Sklearn version: {sklearn.__version__}", flush=True)
except Exception as e:
    print(f"Error sklearn: {e}", flush=True)
