import sys
print("Importing tqdm...", flush=True)
try:
    import tqdm
    print(f"tqdm version: {tqdm.__version__}", flush=True)
except Exception as e:
    print(f"Error tqdm: {e}", flush=True)

print("Importing nltk...", flush=True)
try:
    import nltk
    print(f"nltk version: {nltk.__version__}", flush=True)
except Exception as e:
    print(f"Error nltk: {e}", flush=True)

print("Importing filelock...", flush=True)
try:
    import filelock
    print(f"filelock version: {filelock.__version__}", flush=True)
except Exception as e:
    print(f"Error filelock: {e}", flush=True)

print("Importing pandas...", flush=True)
try:
    import pandas
    print(f"pandas version: {pandas.__version__}", flush=True)
except Exception as e:
    print(f"Error pandas: {e}", flush=True)
