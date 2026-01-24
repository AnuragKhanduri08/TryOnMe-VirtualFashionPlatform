import sys
import os
import json

# Add parent directory to path to find ai_modules
sys.path.append(os.path.dirname(os.getcwd()))

print("Starting diagnostics...")

# Test 1: Data Load
try:
    with open("products.json", "r") as f:
        data = json.load(f)
    print(f"SUCCESS: Loaded {len(data)} products.")
except Exception as e:
    print(f"FAIL: Data load failed: {e}")

# Test 2: Engine Import
try:
    from ai_modules.smart_search.engine import SmartSearchEngine
    print("SUCCESS: Imported SmartSearchEngine")
except Exception as e:
    print(f"FAIL: Import failed: {e}")

# Test 3: Engine Init
try:
    print("Attempting to init engine...")
    engine = SmartSearchEngine()
    print("SUCCESS: Initialized SmartSearchEngine")
except Exception as e:
    print(f"FAIL: Engine init failed: {e}")
