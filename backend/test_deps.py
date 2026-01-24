import sys
print("Importing numpy...", flush=True)
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}", flush=True)
except Exception as e:
    print(f"Error numpy: {e}", flush=True)

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
