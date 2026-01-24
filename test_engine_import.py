
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "ai_modules", "smart_search"))
try:
    from engine import SmartSearchEngine
    print("SmartSearchEngine imported successfully.")
    engine = SmartSearchEngine()
    print("SmartSearchEngine initialized successfully.")
except Exception as e:
    print(f"Failed to import/init SmartSearchEngine: {e}")
