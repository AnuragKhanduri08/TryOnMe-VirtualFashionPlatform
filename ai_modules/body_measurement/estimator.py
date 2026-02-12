from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any, Union
from rembg import remove

class BodyMeasurementEstimator:
    def __init__(self, model_name='yolov8n-pose.pt'):
        """
        Initialize YOLOv8 Pose model and Rembg (no init needed for rembg function).
        """
        print(f"Loading YOLOv8 Pose Model: {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded.")

    def estimate_from_image(self, image_input: Union[str, np.ndarray], user_height_cm: float = 170.0) -> Dict[str, Any]:
        """
        Estimate body landmarks and measurements from an image path or numpy array.
        user_height_cm: User provided height in CM (used for calibration if full body visible).
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
        
        # --- Run Segmentation (Rembg) ---
        # Ensure image is valid for Rembg (path or numpy)
        if isinstance(image_input, str):
            image_np = cv2.imread(image_input)
            # rembg expects RGB? No, standard cv2 is BGR. rembg.remove handles input bytes or array.
            # But best to be safe.
        else:
            image_np = image_input.copy()

        # Rembg removal
        # Returns RGBA image with background transparent
        try:
             # If numpy array, rembg expects it.
             # We want a MASK. 'only_mask=True' returns a single channel alpha mask.
             # Note: rembg.remove(data, only_mask=True)
             
             # Convert BGR to RGB for Rembg just in case? Usually it handles it.
             # Let's assume input is BGR (OpenCV standard).
             
             mask = remove(image_np, only_mask=True)
             # mask is now a 2D numpy array (uint8, 0-255)
             
        except Exception as e:
             print(f"Rembg failed: {e}")
             # Create empty mask as fallback
             h, w = image_np.shape[:2]
             mask = np.zeros((h, w), dtype=np.uint8)

        # Calculate measurements with Hybrid Approach
        measurements = self._calculate_measurements(keypoints, mask, user_height_cm)
        
        return {
            "status": "success",
            "keypoints_count": len(keypoints),
            "measurements": measurements,
            "keypoints": keypoints.tolist() 
        }

    def _calculate_measurements(self, keypoints, mask, user_height_cm: float) -> Dict[str, float]:
        """
        Calculate rough body measurements based on landmarks AND segmentation mask.
        """
        h, w = mask.shape
        
        def dist(idx1, idx2):
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                return 0.0
            
            p1 = keypoints[idx1][:2]
            p2 = keypoints[idx2][:2]
            # Check confidence if available (index 2)
            if keypoints[idx1][2] < 0.3 or keypoints[idx2][2] < 0.3:
                return 0.0
            return float(np.linalg.norm(p1 - p2))

        # --- Helper: Measure Width from Mask at specific Y level ---
        def measure_mask_width(y_level_pct, min_x=0, max_x=w):
            # y_level_pct is 0 to 1 relative to image height? No, exact Y pixel.
            y_idx = int(y_level_pct)
            if y_idx < 0 or y_idx >= h:
                return 0.0
            
            # Get the row
            row = mask[y_idx, :]
            # Find indices where mask > threshold (foreground)
            indices = np.where(row > 0.5)[0]
            
            # Filter indices based on ROI
            indices = indices[(indices >= min_x) & (indices <= max_x)]
            
            if len(indices) == 0:
                return 0.0
            
            # Width is Last Index - First Index
            return float(indices[-1] - indices[0])

        # --- 1. Identify Vertical Levels using YOLO Keypoints ---
        # Check for essential keypoints first
        # Shoulders (5,6) and Hips (11,12) are mandatory for any meaningful upper body measurement
        if not (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5 and 
                keypoints[11][2] > 0.5 and keypoints[12][2] > 0.5):
             # Return zeroed measurements if torso is not clearly visible
             return {
                "shoulder_width_cm": 0, "chest_width_cm": 0, "waist_width_cm": 0, "hip_width_cm": 0,
                "torso_length_cm": 0, "estimated_height_cm": 0,
                "chest_circumference_cm": 0, "waist_circumference_cm": 0,
                "shoulder_width_in": 0, "chest_width_in": 0, "waist_width_in": 0, "hip_width_in": 0,
                "torso_length_in": 0, "estimated_height_in": 0,
                "chest_circumference_in": 0, "waist_circumference_in": 0,
                "suggested_size": "N/A",
                "confidence": "None (Torso Not Detected)",
                "method": "N/A"
             }

        l_shoulder = keypoints[5][:2]
        r_shoulder = keypoints[6][:2]
        l_hip = keypoints[11][:2]
        r_hip = keypoints[12][:2]
        
        # Calculate Torso ROI (Horizontal)
        xs = [l_shoulder[0], r_shoulder[0], l_hip[0], r_hip[0]]
        min_kp_x = min(xs)
        max_kp_x = max(xs)
        torso_width_est = max_kp_x - min_kp_x
        
        # Add padding (e.g. 20% on each side) to allow for flesh/clothing but exclude extended arms
        padding = torso_width_est * 0.2
        roi_min_x = max(0, min_kp_x - padding)
        roi_max_x = min(w, max_kp_x + padding)
        
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        
        torso_len_y = abs(avg_hip_y - avg_shoulder_y)
        
        # Define Scan Lines
        # Chest: Approx 20% down from shoulder
        chest_y = avg_shoulder_y + (torso_len_y * 0.20)
        
        # Waist: Approx 70% down from shoulder (narrowest part usually)
        waist_y = avg_shoulder_y + (torso_len_y * 0.70)
        
        # Hip: At the hip joint level
        hip_y = avg_hip_y
        
        # --- 2. Measure Silhouette Widths (Hybrid) ---
        # If mask is valid, use it. Else fallback to keypoints.
        
        # Fallback Keypoint Widths
        kp_shoulder_width = dist(5, 6)
        kp_hip_width = dist(11, 12)
        
        # Silhouette Widths with ROI Constraint
        sil_chest_width = measure_mask_width(chest_y, roi_min_x, roi_max_x)
        sil_waist_width = measure_mask_width(waist_y, roi_min_x, roi_max_x)
        sil_hip_width = measure_mask_width(hip_y, roi_min_x, roi_max_x)
        
        # Validation: If silhouette width is 0 (mask failed), fallback to heuristics
        if sil_chest_width < 10: sil_chest_width = kp_shoulder_width * 0.85 # Fallback
        if sil_waist_width < 10: sil_waist_width = kp_hip_width * 0.85 # Fallback (Reduced from 1.15)
        if sil_hip_width < 10: sil_hip_width = kp_hip_width * 1.4 # Fallback
        
        # Sanity Check: Clamp Chest Width
        max_expected_width = kp_shoulder_width * 1.3
        if sil_chest_width > max_expected_width:
            sil_chest_width = max_expected_width
            
        # Sanity Check: Waist should generally be smaller than Hips and Chest
        # If Waist > Hips, it's likely catching arms or clothes.
        if sil_waist_width > sil_hip_width:
            sil_waist_width = sil_hip_width * 0.9
            
        # If Waist > Chest, it's also suspicious (unless specific body type, but for estimation safety we clamp)
        if sil_waist_width > sil_chest_width:
             sil_waist_width = sil_chest_width * 0.95
            
        # Shoulder Width
        shoulder_width_px = kp_shoulder_width
        
        # Assign Final Pixels
        chest_width_px = sil_chest_width
        waist_width_px = sil_waist_width
        hip_width_px = sil_hip_width
        
        torso_length_px = float(np.linalg.norm((l_shoulder+r_shoulder)/2 - (l_hip+r_hip)/2))

        # --- 3. Height Estimation & Scaling (Same as before) ---
        # ... (Rest of logic is preserved: Full Body vs Partial Body Strategy) ...
        
        # A. Legs
        l_knee = keypoints[13][:2]
        r_knee = keypoints[14][:2]
        l_ankle = keypoints[15][:2]
        r_ankle = keypoints[16][:2]
        has_legs = (keypoints[15][2] > 0.3 and keypoints[16][2] > 0.3)
        
        if has_legs:
            l_upper_leg = float(np.linalg.norm(l_hip - l_knee))
            r_upper_leg = float(np.linalg.norm(r_hip - r_knee))
            l_lower_leg = float(np.linalg.norm(l_knee - l_ankle))
            r_lower_leg = float(np.linalg.norm(r_knee - r_ankle))
            avg_leg_len = ((l_upper_leg + r_upper_leg) / 2) + ((l_lower_leg + r_lower_leg) / 2)
        else:
            avg_leg_len = torso_length_px * (0.48 / 0.30)

        # B. Head
        nose = keypoints[0][:2]
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        nose_to_shoulder = float(np.linalg.norm(nose - mid_shoulder))
        head_height_est_px = nose_to_shoulder * 2.2
        estimated_height_px = head_height_est_px + torso_length_px + avg_leg_len

        # --- Scaling ---
        scale_factor = 0.0
        confidence = "Low"
        method = "Heuristic"
        
        # Check Visibility
        # Nose(0), Shoulders(5,6), Hips(11,12), Knees(13,14), Ankles(15,16)
        has_head = keypoints[0][2] > 0.5
        has_shoulders = (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5)
        has_hips = (keypoints[11][2] > 0.5 and keypoints[12][2] > 0.5)
        has_knees = (keypoints[13][2] > 0.3 and keypoints[14][2] > 0.3)
        has_ankles = (keypoints[15][2] > 0.3 and keypoints[16][2] > 0.3)

        if has_ankles and has_head:
             # Best Case: Full Body Visible (Head to Toe)
             scale_factor = user_height_cm / (estimated_height_px + 1e-6)
             confidence = "High (Full Body Calibration)"
             method = "Height Calibration"
             
        elif has_knees and has_head:
             # Partial: Head to Knees visible. Extrapolate lower legs.
             # Anthropometry: Knee height is approx 28.5% of total height (or 0.285 * H)
             # So Head-to-Knee is (1 - 0.285) = 0.715 of Height
             
             # Calculate pixel distance from Top of Head to Knees
             # Top of Head approx: NoseY - (Nose-Shoulder)*1.2
             head_top_y = nose[1] - (nose_to_shoulder * 1.0)
             avg_knee_y = (l_knee[1] + r_knee[1]) / 2
             visible_body_px = abs(avg_knee_y - head_top_y)
             
             # Estimated Total Height = Visible / 0.72
             est_total_height_px = visible_body_px / 0.72
             scale_factor = user_height_cm / (est_total_height_px + 1e-6)
             
             confidence = "Medium-High (Knees Visible)"
             method = "Knee-Height Ratio"
             
        elif has_hips and has_shoulders:
             # Torso Only Visible. 
             # Torso Length (Shoulder to Hip) is approx 30% of height (0.30)
             # But this varies. Let's use 0.30 as standard.
             
             # Refined Torso Ratio:
             scale_factor = (user_height_cm * 0.30) / (torso_length_px + 1e-6)
             confidence = "Medium (Torso Ratio)"
             method = "Torso Ratio"
             
        elif has_head:
             target_neck_head_cm = user_height_cm * 0.14
             scale_factor = target_neck_head_cm / (nose_to_shoulder + 1e-6)
             confidence = "Low (Head Ratio)"
             method = "Head Ratio"
        else:
             target_shoulder_cm = user_height_cm * 0.24
             scale_factor = target_shoulder_cm / (shoulder_width_px + 1e-6)
             confidence = "Very Low (Shoulder Ratio)"
             method = "Shoulder Ratio"
        
        # Final CMs
        chest_width_cm = chest_width_px * scale_factor
        waist_width_cm = waist_width_px * scale_factor
        shoulder_width_cm = shoulder_width_px * scale_factor
        hip_width_cm = hip_width_px * scale_factor
        
        # Circumferences (More accurate now with silhouette)
        # Since we have "Outer Width", circumference is closer to Width * 2 + Depth * 2.
        # Using 2.3 multiplier to be more conservative and avoid overestimation (XXL bias).
        chest_circ = chest_width_cm * 2.3
        waist_circ = waist_width_cm * 2.2 # Reduced from 2.3 to account for flatter waist shape
        
        # Convert to Inches
        def to_in(cm_val):
             return round(cm_val / 2.54, 1) if cm_val is not None else None

        measurements = {
            "shoulder_width_cm": round(shoulder_width_cm, 1),
            "chest_width_cm": round(chest_width_cm, 1),
            "waist_width_cm": round(waist_width_cm, 1),
            "hip_width_cm": round(hip_width_cm, 1),
            "torso_length_cm": round(torso_length_px * scale_factor, 1),
            "estimated_height_cm": round(estimated_height_px * scale_factor, 1) if (has_ankles or has_knees) else None,
            "chest_circumference_cm": round(chest_circ, 1),
            "waist_circumference_cm": round(waist_circ, 1),
            
            # Inches
            "shoulder_width_in": to_in(shoulder_width_cm),
            "chest_width_in": to_in(chest_width_cm),
            "waist_width_in": to_in(waist_width_cm),
            "hip_width_in": to_in(hip_width_cm),
            "torso_length_in": to_in(torso_length_px * scale_factor),
            "estimated_height_in": to_in(estimated_height_px * scale_factor) if (has_ankles or has_knees) else None,
            "chest_circumference_in": to_in(chest_circ),
            "waist_circumference_in": to_in(waist_circ),

            "suggested_size": "M",
            "confidence": confidence,
            "method": method
        }
        
        cc = measurements["chest_circumference_cm"]
        if cc < 88: measurements["suggested_size"] = "XS"
        elif cc < 96: measurements["suggested_size"] = "S"
        elif cc < 104: measurements["suggested_size"] = "M"
        elif cc < 112: measurements["suggested_size"] = "L"
        elif cc < 124: measurements["suggested_size"] = "XL"
        else: measurements["suggested_size"] = "XXL"
        
        return {
            "shoulder_width_pixels": shoulder_width_px,
            "chest_width_pixels": chest_width_px,
            "waist_width_pixels": waist_width_px,
            "hip_width_pixels": hip_width_px,
            "torso_length_pixels": torso_length_px,
            "estimated_height_pixels": estimated_height_px,
            "cm_estimates": measurements
        }

if __name__ == "__main__":
    # Simple test if run directly
    print("BodyMeasurementEstimator initialized.")
