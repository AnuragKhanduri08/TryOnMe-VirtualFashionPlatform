from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any, Union

class BodyMeasurementEstimator:
    def __init__(self, model_name='yolov8n-pose.pt'):
        """
        Initialize YOLOv8 Pose model.
        """
        print(f"Loading YOLOv8 Pose Model: {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded.")

    def estimate_from_image(self, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Estimate body landmarks and measurements from an image path or numpy array.
        """
        # YOLOv8 handles path or array directly
        results = self.model(image_input)
        
        # Process first result
        result = results[0]
        
        # Check if any detections
        if result.boxes is None or len(result.boxes) == 0:
             return {"status": "error", "message": "No person detected"}
        
        if result.keypoints is None or not result.keypoints.has_visible:
             return {"status": "error", "message": "No pose detected"}

        # Keypoints: (N, 17, 3) -> x, y, conf
        # We take the first detected person
        keypoints = result.keypoints.data[0].cpu().numpy()
        
        # Calculate measurements
        measurements = self._calculate_measurements(keypoints)
        
        return {
            "status": "success",
            "keypoints_count": len(keypoints),
            "measurements": measurements,
            "keypoints": keypoints.tolist() # Return raw keypoints for other modules (e.g. Try-On)
        }

    def _calculate_measurements(self, keypoints) -> Dict[str, float]:
        """
        Calculate rough body measurements based on landmarks.
        YOLO Keypoints map (COCO format):
        0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
        5: Left Shoulder, 6: Right Shoulder
        7: Left Elbow, 8: Right Elbow
        9: Left Wrist, 10: Right Wrist
        11: Left Hip, 12: Right Hip
        13: Left Knee, 14: Right Knee
        15: Left Ankle, 16: Right Ankle
        """
        
        def dist(idx1, idx2):
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                return 0.0
            
            p1 = keypoints[idx1][:2]
            p2 = keypoints[idx2][:2]
            # Check confidence if available (index 2)
            if keypoints[idx1][2] < 0.5 or keypoints[idx2][2] < 0.5:
                return 0.0
            return float(np.linalg.norm(p1 - p2))

        shoulder_width = dist(5, 6)
        hip_width = dist(11, 12)
        
        # Torso Length: Midpoint shoulder to Midpoint hip
        l_shoulder = keypoints[5][:2]
        r_shoulder = keypoints[6][:2]
        l_hip = keypoints[11][:2]
        r_hip = keypoints[12][:2]
        
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (l_hip + r_hip) / 2
        torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))

        # Estimate Total Height (Segments method)
        # 1. Head (approx 2x Nose-to-Shoulder)
        nose = keypoints[0][:2]
        head_neck_dist = float(np.linalg.norm(nose - mid_shoulder))
        
        # 2. Legs (Average of left/right)
        l_knee = keypoints[13][:2]
        r_knee = keypoints[14][:2]
        l_ankle = keypoints[15][:2]
        r_ankle = keypoints[16][:2]
        
        # Upper Leg (Hip to Knee)
        l_upper_leg = float(np.linalg.norm(l_hip - l_knee))
        r_upper_leg = float(np.linalg.norm(r_hip - r_knee))
        avg_upper_leg = (l_upper_leg + r_upper_leg) / 2
        
        # Lower Leg (Knee to Ankle)
        l_lower_leg = float(np.linalg.norm(l_knee - l_ankle))
        r_lower_leg = float(np.linalg.norm(r_knee - r_ankle))
        avg_lower_leg = (l_lower_leg + r_lower_leg) / 2
        
        # Total Height = Head(approx) + Torso + Legs + Foot(approx)
        # Head top to Nose is approx same as Nose to Neck (Head/Neck dist)
        # So Head segment ~ 2 * head_neck_dist
        # Foot height approx 10% of lower leg
        estimated_height = (head_neck_dist * 2.0) + torso_length + avg_upper_leg + avg_lower_leg + (avg_lower_leg * 0.15)

        return {
            "shoulder_width_pixels": shoulder_width,
            "hip_width_pixels": hip_width,
            "torso_length_pixels": torso_length,
            "estimated_height_pixels": estimated_height,
            "shoulder_hip_ratio": float(shoulder_width / (hip_width + 1e-6)) if hip_width > 0 else 0.0
        }

if __name__ == "__main__":
    # Simple test if run directly
    print("BodyMeasurementEstimator initialized.")
