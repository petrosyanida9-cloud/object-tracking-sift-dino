import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


device = "cuda" if torch.cuda.is_available() else "cpu"
target_image_path = "data/inputs/target.jpg"
MIN_MATCH_COUNT = 7
LOWE_RATIO = 0.75


input_folder   = "data/inputs/images"
output_folder  = "data/outputs/output_accurate"
matches_folder = "data/outputs/matching_visuals_0.75"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(matches_folder, exist_ok=True)


sift = cv2.SIFT_create()
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
target_img_color = cv2.imread(target_image_path)
if target_img is None:
    print(f"[Warning] Target image not found at {target_image_path}")
    print("Please place 'target.jpg' inside 'data/inputs/' to test.")
    
    os.makedirs(input_folder, exist_ok=True)
    exit(0)

kp_target, des_target = sift.detectAndCompute(target_img, None)


model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

text_prompt = "small object . tiny structure . central building ."
BOX_THRESHOLD = 0.18

print(f"Processing image collection on {device}...")


if not os.path.exists(input_folder) or len(os.listdir(input_folder)) == 0:
    os.makedirs(input_folder, exist_ok=True)
    print(f"[Info] Input folder '{input_folder}' is empty. Place test photos there.")
    exit(0)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.JPEG')):
        image_path = os.path.join(input_folder, filename)
        try:
            raw_image = Image.open(image_path).convert("RGB")
            cv_image = cv2.imread(image_path)
            inputs = processor(images=raw_image, text=text_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            w, h = raw_image.size
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, target_sizes=[(h, w)]
            )[0]

            boxes = results["boxes"][results["scores"] > BOX_THRESHOLD].cpu().numpy()

            best_match_data = None
            max_good_matches = 0

            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                crop = cv_image[y1:y2, x1:x2]
                if crop.size == 0: continue

                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                kp_crop, des_crop = sift.detectAndCompute(gray_crop, None)

                if des_crop is None or len(des_crop) < 2: continue

                matches = flann.knnMatch(des_target, des_crop, k=2)
                good_matches = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

                if len(good_matches) > max_good_matches and len(good_matches) >= MIN_MATCH_COUNT:
                    max_good_matches = len(good_matches)
                    best_match_data = {
                        "box": box,
                        "kp_crop": kp_crop,
                        "good_matches": good_matches,
                        "crop": crop
                    }

            if best_match_data:
                
                detections = sv.Detections(
                    xyxy=np.array([best_match_data["box"]]),
                    confidence=np.array([1.0]),
                    class_id=np.array([0])
                )
                box_annotator = sv.BoxAnnotator(
                    color=sv.Color.from_hex("#FFFF00"), 
                    thickness=2
                )
                annotated_frame = box_annotator.annotate(scene=cv_image.copy(), detections=detections)
                cv2.imwrite(os.path.join(output_folder, f"det_{filename}"), annotated_frame)

                
                match_img = cv2.drawMatches(
                    target_img_color, kp_target,
                    best_match_data["crop"], best_match_data["kp_crop"],
                    best_match_data["good_matches"], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                match_output_path = os.path.join(matches_folder, f"match_{filename}")
                cv2.imwrite(match_output_path, match_img)

                print(f" -> {filename}: Verified with {max_good_matches} matches.")
            else:
                print(f" -> {filename}: No match.")

        except Exception as e:
            print(f"Error on {filename}: {e}")

print("Done! Check 'data/outputs/' directory for results.")
