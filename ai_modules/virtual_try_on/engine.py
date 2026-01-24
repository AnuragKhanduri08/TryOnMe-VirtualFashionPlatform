import numpy as np
import cv2
from typing import Dict, Any, Union, Optional
from PIL import Image
import os
import tempfile
from gradio_client import Client, handle_file
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False
    print(f"Warning: rembg not installed. {e}")

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: transformers/torch not installed or disabled. {e}")

class VirtualTryOnEngine:
    def __init__(self):
        """
        Initialize Virtual Try-On Engine.
        """
        print("Initializing Virtual Try-On Engine...")
        self.client = None
        self.current_token = None
        
        # Lazy loading for models
        self.segformer_model = None
        self.segformer_processor = None
        
        # Initial connection attempt (anonymous)
        self.connect_to_cloud()

    def _get_segformer_model(self):
        """Load Segformer model lazily."""
        if self.segformer_model is not None:
            return self.segformer_processor, self.segformer_model
        
        try:
            print("Loading Segformer model for cloth segmentation...")
            self.segformer_processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            return self.segformer_processor, self.segformer_model
        except Exception as e:
            print(f"Failed to load Segformer: {e}")
            return None, None

    def connect_to_cloud(self, token: str = None):
        if self.client and self.current_token == token:
            return # Already connected with this token
            
        try:
            print(f"Connecting to IDM-VTON Space (Token: {'Yes' if token else 'No'})...")
            # We use yisol/IDM-VTON as it is SOTA.
            if token:
                self.client = Client("yisol/IDM-VTON", hf_token=token)
            else:
                self.client = Client("yisol/IDM-VTON")
                
            self.current_token = token
            print("Connected to Hugging Face Space: yisol/IDM-VTON")
        except Exception as e:
            print(f"Failed to connect to HF Space: {e}. Cloud try-on will not work.")
            self.client = None

    def try_on_cloud(self, person_image: np.ndarray, cloth_image: np.ndarray, hf_token: str = None) -> Union[np.ndarray, None]:
        """
        Perform high-quality virtual try-on using Cloud API (IDM-VTON).
        """
        # Re-connect if client is missing or token changed
        if not self.client or (hf_token and hf_token != self.current_token):
            self.connect_to_cloud(hf_token)
            
        if not self.client:
            print("Cloud client not initialized (Connection failed).")
            # Try OOTDiffusion immediately if IDM-VTON connection failed
            pass 

        # Store original dimensions for restoration
        orig_h, orig_w = person_image.shape[:2]

        try:
            if not self.client:
                raise RuntimeError("IDM-VTON Client could not be initialized.")

            # 1. Save images to temp files (Gradio client needs file paths)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                
                cv2.imwrite(f1.name, person_image)
                cv2.imwrite(f2.name, cloth_image)
                
                person_path = f1.name
                cloth_path = f2.name

            print("Sending request to IDM-VTON Cloud API...")
            
            # The API signature for yisol/IDM-VTON:
            # predict(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, api_name="/tryon")
            
            result_path = self.client.predict(
                {"background": handle_file(person_path), "layers": [], "composite": None}, # dict
                handle_file(cloth_path), # garm_img
                "A cool fashion item",   # garment_des
                True,                    # is_checked (auto-mask)
                False,                   # is_checked_crop
                30,                      # denoise_steps
                42,                      # seed
                api_name="/tryon"
            )
            
            print(f"Cloud API Result: {result_path}")
            
            # The API returns (image, mask) usually. We take the image.
            output_file = result_path[0] if isinstance(result_path, (list, tuple)) else result_path
            
            # Read back result
            if output_file and os.path.exists(output_file):
                result_img = cv2.imread(output_file)
                if result_img is None:
                    print("Error: Could not read result image from cloud.")
                    return None
                
                # Restore original resolution
                if result_img.shape[0] != orig_h or result_img.shape[1] != orig_w:
                    print(f"Restoring resolution: {result_img.shape[1]}x{result_img.shape[0]} -> {orig_w}x{orig_h}")
                    result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                
                return result_img
                
        except Exception as e:
            error_msg = f"Cloud Try-On Failed with Error: {e}"
            print(error_msg)
            # Log detailed error
            with open("engine_error.log", "a") as err_log:
                err_log.write(error_msg + "\n")
            
            # Check if temp files exist (they might not if exception happened early)
            if 'person_path' not in locals():
                # Create them now for fallback
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
                     tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                    cv2.imwrite(f1.name, person_image)
                    cv2.imwrite(f2.name, cloth_image)
                    person_path = f1.name
                    cloth_path = f2.name

            # Fallback to OOTDiffusion if IDM-VTON fails
            print("Attempting fallback to OOTDiffusion (Half Body)...")
            res = self.try_on_ootd_hd_fallback(person_path, cloth_path)
            if res is not None:
                # Restore resolution for fallback
                if res.shape[0] != orig_h or res.shape[1] != orig_w:
                    print(f"Restoring resolution (Fallback): {res.shape[1]}x{res.shape[0]} -> {orig_w}x{orig_h}")
                    res = cv2.resize(res, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                return res
            
            print("Attempting fallback to OOTDiffusion (Full Body)...")
            res = self.try_on_ootd_dc_fallback(person_path, cloth_path)
            if res is not None:
                # Restore resolution for fallback
                if res.shape[0] != orig_h or res.shape[1] != orig_w:
                    print(f"Restoring resolution (Fallback 2): {res.shape[1]}x{res.shape[0]} -> {orig_w}x{orig_h}")
                    res = cv2.resize(res, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                return res

        finally:
            # Cleanup temp files
            if 'person_path' in locals() and os.path.exists(person_path): os.unlink(person_path)
            if 'cloth_path' in locals() and os.path.exists(cloth_path): os.unlink(cloth_path)

        return None

    def try_on_ootd_hd_fallback(self, person_path, cloth_path):
        try:
            client = Client("levihsu/OOTDiffusion")
            print("Connected to OOTDiffusion (HD) fallback.")
            
            result_path = client.predict(
                handle_file(person_path), # vton_img
                handle_file(cloth_path),  # garm_img
                1,                        # n_samples
                20,                       # n_steps
                2.0,                      # image_scale
                -1,                       # seed
                api_name="/process_hd"
            )
            return self._process_ootd_result(result_path)
        except Exception as e:
            print(f"Fallback OOTDiffusion (HD) failed: {e}")
            return None

    def try_on_ootd_dc_fallback(self, person_path, cloth_path):
        try:
            client = Client("levihsu/OOTDiffusion")
            print("Connected to OOTDiffusion (DC) fallback.")
            
            # /process_dc takes 'category' as 3rd arg
            result_path = client.predict(
                handle_file(person_path), # vton_img
                handle_file(cloth_path),  # garm_img
                "Upper-body",             # category
                1,                        # n_samples
                20,                       # n_steps
                2.0,                      # image_scale
                -1,                       # seed
                api_name="/process_dc"
            )
            return self._process_ootd_result(result_path)
        except Exception as e:
            print(f"Fallback OOTDiffusion (DC) failed: {e}")
            return None

    def _process_ootd_result(self, result_path):
        # OOTDiffusion returns a Gallery (list of dicts)
        output_file = None
        if isinstance(result_path, list) and len(result_path) > 0:
            item = result_path[0]
            if isinstance(item, dict) and 'image' in item:
                output_file = item['image']
            elif isinstance(item, str):
                output_file = item
        elif isinstance(result_path, str):
            output_file = result_path
            
        if output_file and os.path.exists(output_file):
            return cv2.imread(output_file)
        return None

    def segment_cloth(self, cloth_image: np.ndarray, cloth_type: str = "upper") -> np.ndarray:
        """
        Extract cloth from image using Semantic Segmentation (Segformer) or Background Removal (rembg).
        cloth_type: "upper", "lower", or "full" (default: "upper" - prioritizes shirts/dresses)
        """
        print(f"Segmenting cloth (Type: {cloth_type})...")
        
        # 1. Try Segformer (Best for removing model body/skin)
        if TRANSFORMERS_AVAILABLE:
             processor, model = self._get_segformer_model()
             if processor and model:
                 try:
                     print("Using Segformer for segmentation...")
                     # Convert to RGB PIL
                     if len(cloth_image.shape) == 2:
                         rgb = cv2.cvtColor(cloth_image, cv2.COLOR_GRAY2RGB)
                     elif cloth_image.shape[2] == 3:
                         rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
                     elif cloth_image.shape[2] == 4:
                         rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGRA2RGB)
                     else:
                         rgb = cloth_image # Assume RGB if unknown
                     
                     pil_img = Image.fromarray(rgb)
                     inputs = processor(images=pil_img, return_tensors="pt")
                     
                     with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                     
                     # Upsample logits to original size
                     upsampled_logits = torch.nn.functional.interpolate(
                         logits,
                         size=pil_img.size[::-1], # H, W
                         mode="bilinear",
                         align_corners=False,
                     )
                     
                     pred_seg = upsampled_logits.argmax(dim=1)[0]
                     
                     # Labels for "mattmdjaga/segformer_b2_clothes":
                     # 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes, 5: Skirt, 
                     # 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe, 11: Face, 
                     # 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf
                     
                     labels_found = torch.unique(pred_seg).tolist()
                     print(f"Segformer found labels: {labels_found}")
                     
                     has_upper = 4 in labels_found
                     has_dress = 7 in labels_found
                     has_pants = 6 in labels_found
                     has_skirt = 5 in labels_found
                     
                     mask = None
                     
                     if cloth_type == "upper":
                         if has_upper or has_dress:
                             print("Keeping Upper-clothes (4) and Dress (7). Excluding arms/face/pants.")
                             mask = (pred_seg == 4) | (pred_seg == 7)
                         elif has_pants or has_skirt:
                             print("No upper body found, but pants/skirt found. Keeping them as fallback.")
                             mask = (pred_seg == 6) | (pred_seg == 5)
                     elif cloth_type == "lower":
                         if has_pants or has_skirt:
                             print("Keeping Pants (6) and Skirt (5).")
                             mask = (pred_seg == 6) | (pred_seg == 5)
                     
                     # Fallback if specific type not found or general request
                     if mask is None:
                         print("Using generic cloth mask (Upper+Lower+Dress).")
                         mask = (pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7)
                     
                     # Convert mask to numpy
                     mask_np = mask.numpy().astype(np.uint8) * 255

                     
                     # Check if mask is not empty
                     if np.count_nonzero(mask_np) > 0:
                         print("Segformer: Cloth detected.")
                         # Apply mask (Alpha channel)
                         b, g, r = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                         rgba = cv2.merge([b, g, r, mask_np])
                         
                         # Crop to content
                         coords = cv2.findNonZero(mask_np)
                         x, y, w, h = cv2.boundingRect(coords)
                         cropped = rgba[y:y+h, x:x+w]
                         return cropped
                     else:
                         print("Segformer: No cloth labels found. Falling back to rembg.")
                 except Exception as e:
                     print(f"Segformer failed: {e}")

        # 2. Fallback to rembg (Background removal only)
        if REMBG_AVAILABLE:
            try:
                # Convert cv2 (BGR) to PIL (RGB)
                if len(cloth_image.shape) == 2: # Grayscale
                    rgb = cv2.cvtColor(cloth_image, cv2.COLOR_GRAY2RGB)
                elif cloth_image.shape[2] == 3: # BGR
                    rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
                elif cloth_image.shape[2] == 4: # BGRA
                    rgb = cv2.cvtColor(cloth_image, cv2.COLOR_BGRA2RGBA)
                else:
                    return cloth_image # Unknown format
                pil_img = Image.fromarray(rgb)
                # Remove background
                output = remove(pil_img)
                # Convert back to cv2 (BGRA)
                out_np = np.array(output)
                out_bgra = cv2.cvtColor(out_np, cv2.COLOR_RGBA2BGRA)
                return out_bgra
            except Exception as e:
                print(f"Background removal failed: {e}")
                return cloth_image
        
        return cloth_image

    def try_on(self, person_image: np.ndarray, cloth_image: np.ndarray, keypoints: list, adjustments: dict = None, use_cloud: bool = False, hf_token: str = None) -> np.ndarray:
        """
        Perform virtual try-on.
        """
        if use_cloud:
            print("Attempting Cloud Try-On...")
            try:
                cloud_result = self.try_on_cloud(person_image, cloth_image, hf_token)
                if cloud_result is not None:
                    return cloud_result
                
                # If we get here, cloud models returned None (failed)
                print("Cloud models failed. Raising error to notify user.")
                raise RuntimeError("Cloud services are currently busy or unavailable. Please try again later or add a Hugging Face Token.")
                
            except Exception as e:
                print(f"Cloud try-on error: {e}")
                # We do NOT fallback to local if user specifically asked for cloud.
                # This prevents user from seeing a "bad" local result and thinking it's the cloud result.
                raise e
            
        try:
            # 1. Basic Validation
            if person_image is None or cloth_image is None:
                raise ValueError("Images cannot be None")
                
            # Default adjustments
            adj_scale = 1.0
            adj_x = 0
            adj_y = 0
            
            if adjustments:
                adj_scale = float(adjustments.get('scale', 1.0))
                adj_x = float(adjustments.get('x', 0))
                adj_y = float(adjustments.get('y', 0))

            # 2. Extract Keypoints for Upper Body
            # COCO: 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip
            kp = np.array(keypoints)
            
            # Ensure we have valid keypoints
            if len(kp) < 13:
                 print("Insufficient keypoints.")
                 return person_image

            l_shoulder = kp[5][:2]
            r_shoulder = kp[6][:2]
            l_hip = kp[11][:2]
            r_hip = kp[12][:2]
            
            # Check confidence
            if kp[5][2] < 0.3 or kp[6][2] < 0.3:
                 print("Low confidence on shoulders, returning original image.")
                 return person_image
            
            # If hips are low confidence, estimate them from shoulders
            if kp[11][2] < 0.3 or kp[12][2] < 0.3:
                # Estimate hips: Down from shoulders by 1.5x shoulder width
                shoulder_width_est = np.linalg.norm(l_shoulder - r_shoulder)
                # Perpendicular vector estimation could be better, but simple vertical drop is okay for fallback
                l_hip = [l_shoulder[0], l_shoulder[1] + shoulder_width_est * 1.5]
                r_hip = [r_shoulder[0], r_shoulder[1] + shoulder_width_est * 1.5]

            # 3. Prepare Cloth Image
            # Check for Alpha Channel
            # AUTO-SEGMENTATION: Always attempt segmentation to ensure we get just the cloth,
            # even if the user uploads a PNG that might still contain body parts.
            # Only skip if we are 100% sure it's already segmented (hard to tell, so we prefer running Segformer).
            
            print(f"Cloth image shape: {cloth_image.shape}. Running smart segmentation...")
            try:
                # Force segmentation. The method handles 3 or 4 channels internally.
                cloth_image = self.segment_cloth(cloth_image)
            except Exception as e:
                print(f"Segmentation wrapper failed: {e}")
                # Fallback handled inside segment_cloth, but if wrapper fails, continue with original
                pass

            has_alpha = False
            if cloth_image.shape[2] == 4:
                has_alpha = True
                b, g, r, a = cv2.split(cloth_image)
                cloth_rgb = cv2.merge((b, g, r))
                cloth_mask = a
            else:
                cloth_rgb = cloth_image
                gray_cloth = cv2.cvtColor(cloth_rgb, cv2.COLOR_BGR2GRAY)
                _, cloth_mask = cv2.threshold(gray_cloth, 240, 255, cv2.THRESH_BINARY_INV)

            # 4. Enhanced Local Warping (Perspective Transform)
            # Define Source Points (Corners of the cloth image)
            h_cloth, w_cloth = cloth_rgb.shape[:2]
            
            # REMOVE HEURISTIC CROP IF SEGMENTATION IS ACTIVE
            # If we used segmentation, the image should already be tight-ish or just the shirt.
            # But rembg keeps the original size, just makes background transparent.
            # So we should find the bounding box of the non-transparent part!
            
            if has_alpha:
                # Find bounding box of mask
                coords = cv2.findNonZero(cloth_mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    print(f"Cropping to content: x={x}, y={y}, w={w}, h={h}")
                    cloth_rgb = cloth_rgb[y:y+h, x:x+w]
                    cloth_mask = cloth_mask[y:y+h, x:x+w]
                    h_cloth, w_cloth = cloth_rgb.shape[:2]
            
            # (Old Heuristic Crop Logic Removed/Replaced by Alpha Crop)

            src_pts = np.float32([
                [0, 0],             # Top-Left
                [w_cloth, 0],       # Top-Right
                [0, h_cloth],       # Bottom-Left
                [w_cloth, h_cloth]  # Bottom-Right
            ])

            # Define Target Points on Person
            # We want the shirt to cover shoulders and hips, plus some padding (scale)
            
            # Calculate vectors
            shoulder_vec = r_shoulder - l_shoulder
            shoulder_width = np.linalg.norm(shoulder_vec)
            
            # Vertical vector (approximate)
            # Average of L-Shoulder->L-Hip and R-Shoulder->R-Hip
            left_side_vec = l_hip - l_shoulder
            right_side_vec = r_hip - r_shoulder
            
            # Scaling factors (Tunable)
            # Make shirt 20% wider than shoulder points
            width_pad = shoulder_width * (0.2 + (adj_scale - 1.0)) 
            
            # Move points outward along the shoulder vector
            # Normalize shoulder vector
            if shoulder_width > 0:
                u_shoulder = shoulder_vec / shoulder_width
            else:
                u_shoulder = np.array([1.0, 0.0])

            # Top Left Target
            # Shift Left by width_pad, Shift Up by 15% of torso length
            # Note: We use -u_shoulder to go Left
            # Vertical shift: We assume "Up" is negative Y.
            # But better to use the side vectors.
            
            tl_target = l_shoulder - (u_shoulder * width_pad) - (left_side_vec * 0.15)
            tr_target = r_shoulder + (u_shoulder * width_pad) - (right_side_vec * 0.15)
            
            # Bottom targets
            # We want the shirt to go slightly below hips
            bl_target = l_hip - (u_shoulder * width_pad) + (left_side_vec * 0.1)
            br_target = r_hip + (u_shoulder * width_pad) + (right_side_vec * 0.1)
            
            # Apply X/Y Adjustments (Global Shift)
            # adj_x, adj_y are relative to shoulder_width and torso_length
            shift_vec = np.array([adj_x * shoulder_width, adj_y * np.linalg.norm(left_side_vec)])
            
            dst_pts = np.float32([
                tl_target + shift_vec,
                tr_target + shift_vec,
                bl_target + shift_vec,
                br_target + shift_vec
            ])

            # Calculate Perspective Transform Matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Warp Image and Mask
            h_person, w_person = person_image.shape[:2]
            warped_cloth = cv2.warpPerspective(cloth_rgb, M, (w_person, h_person))
            warped_mask = cv2.warpPerspective(cloth_mask, M, (w_person, h_person))

            # 5. Alpha Blending
            # Refine mask to remove jagged edges
            # warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
            
            # Convert to float for blending
            mask_alpha = warped_mask.astype(float) / 255.0
            mask_alpha = np.dstack([mask_alpha] * 3) # 3 channels
            
            person_float = person_image.astype(float)
            cloth_float = warped_cloth.astype(float)
            
            # Blend: Output = Cloth * Alpha + Person * (1 - Alpha)
            out_float = (cloth_float * mask_alpha) + (person_float * (1.0 - mask_alpha))
            
            result_image = out_float.astype(np.uint8)
            
            # DEBUG: Draw Keypoints on result to debug placement
            # L-Shoulder (Red), R-Shoulder (Green), L-Hip (Blue), R-Hip (Yellow)
            cv2.circle(result_image, (int(l_shoulder[0]), int(l_shoulder[1])), 5, (0, 0, 255), -1)
            cv2.circle(result_image, (int(r_shoulder[0]), int(r_shoulder[1])), 5, (0, 255, 0), -1)
            cv2.circle(result_image, (int(l_hip[0]), int(l_hip[1])), 5, (255, 0, 0), -1)
            cv2.circle(result_image, (int(r_hip[0]), int(r_hip[1])), 5, (0, 255, 255), -1)
            
            return result_image

        except Exception as e:
            print(f"Error in Try-On: {e}")
            import traceback
            traceback.print_exc()
            return person_image # Return original on error

if __name__ == "__main__":
    engine = VirtualTryOnEngine()
    print("Virtual Try-On Engine initialized.")
