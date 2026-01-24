import sys
import os
import traceback

print("Starting debug...")
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Added to sys.path")
    
    import fastapi
    print("Imported fastapi")
    
    from ai_modules.smart_search.engine import SmartSearchEngine
    print("Imported SmartSearchEngine")
    
    from ai_modules.body_measurement.estimator import BodyMeasurementEstimator
    print("Imported BodyMeasurementEstimator")
    
    from ai_modules.virtual_try_on.engine import VirtualTryOnEngine
    print("Imported VirtualTryOnEngine")
    
    print("All imports successful")
except Exception as e:
    print("Error during imports:")
    traceback.print_exc()
