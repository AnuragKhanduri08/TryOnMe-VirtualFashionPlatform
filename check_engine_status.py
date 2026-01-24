import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.smart_search.engine import SmartSearchEngine

engine = SmartSearchEngine()
if engine.model is not None:
    print("Engine Model Loaded Successfully")
else:
    print("Engine Model Failed to Load")
