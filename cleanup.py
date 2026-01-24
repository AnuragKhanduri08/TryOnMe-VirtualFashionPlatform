
import shutil
import os

path = r"C:\Users\khand\AppData\Roaming\Python\Python312\site-packages\~umpy"
if os.path.exists(path):
    print(f"Removing {path}")
    try:
        shutil.rmtree(path)
        print("Removed.")
    except Exception as e:
        print(f"Failed to remove: {e}")
else:
    print("Path does not exist.")
