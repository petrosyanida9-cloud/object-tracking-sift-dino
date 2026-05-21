

import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = "cuda" if torch.cuda.is_available() else "cpu"
target_image_path = "/content/target.jpg"
input_video_path  = "/content/video.mp4"
output_video_path        = "output_detected_video.mp4"
output_matches_video_path = "output_matches_video.mp4"

MIN_MATCH_COUNT = 7
LOWE_RATIO      = 0.75
DINO_RETRY_FRAMES = 30   

sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5),
    dict(checks=50))

target_img_color = cv2.imread(target_image_path)
if target_img_color is None:
    raise FileNotFoundError(f"Target image not found: {target_image_path}")

orig_target_color = target_img_color.copy()   
target_img        = cv2.cvtColor(target_img_color, cv2.COLOR_BGR2GRAY)
kp_target, des_target = sift.detectAndCompute(target_img, None)

TARGET_H, TARGET_W = target_img_color.shape[:2]  


model_id  = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

text_prompt   = "small object . tiny structure . central building ."
BOX_THRESHOLD = 0.18
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_det   = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

MATCH_W = TARGET_W + width
MATCH_H = max(TARGET_H, height)
out_match = cv2.VideoWriter(output_matches_video_path, fourcc, fps, (MATCH_W, MATCH_H))

def run_dino_and_verify(frame, kp_t, des_t):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    inputs = processor(images=pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[(height, width)]
    )[0]

    boxes = results["boxes"][results["scores"] > BOX_THRESHOLD].cpu().numpy()

    best, max_good = None, 0
    for box in boxes:
        x1, y1, x2, y2 = np.clip(box.astype(int),
                                  [0, 0, 0, 0],
                                  [width, height, width, height])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        if des_c is None or len(des_c) < 2:
            continue
        matches    = flann.knnMatch(des_t, des_c, k=2)
        good       = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
        if len(good) >= MIN_MATCH_COUNT and len(good) > max_good:
            max_good = len(good)
            best = dict(box=box.astype(float), kp_crop=kp_c,
                        good_matches=good, crop=crop)
    return best
def blank_match_frame():
    return np.zeros((MATCH_H, MATCH_W, 3), dtype=np.uint8)
def build_match_frame(t_color, kp_t, crop, kp_c, good):
    img = cv2.drawMatches(
        t_color, kp_t, crop, kp_c, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    h, w = img.shape[:2]
    canvas = np.zeros((MATCH_H, MATCH_W, 3), dtype=np.uint8)
    canvas[:min(h, MATCH_H), :min(w, MATCH_W)] = img[:MATCH_H, :MATCH_W]
    return canvas

def annotate(frame, box_xyxy, hex_color):
    det = sv.Detections(
        xyxy=np.array([box_xyxy]),
        confidence=np.array([1.0]),
        class_id=np.array([0])
    )
    annotator = sv.BoxAnnotator(
        color=sv.Color.from_hex(hex_color),
        thickness=2,
        color_lookup=sv.ColorLookup.INDEX
    )
    return annotator.annotate(scene=frame.copy(), detections=det)

tracker             = None
tracker_initialized = False
frame_count         = 0
last_tracked_box    = None   
last_tracked_crop   = None

draw_target_color = orig_target_color.copy()
draw_kp_target    = kp_target
draw_des_target   = des_target

print(f"Processing on {device} …")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            wrote_det   = False
            wrote_match = False

            run_dino = (frame_count == 1) or \
                       (not tracker_initialized and frame_count % DINO_RETRY_FRAMES == 0)

            if run_dino:
                best = run_dino_and_verify(frame, draw_kp_target, draw_des_target)
                if best:
                    last_tracked_box  = best["box"]
                    last_tracked_crop = best["crop"]

                    x1, y1, x2, y2 = last_tracked_box.astype(int)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                    tracker_initialized = True

                    out_det.write(annotate(frame, last_tracked_box, "#FFFF00"))
                    out_match.write(build_match_frame(
                        draw_target_color, draw_kp_target,
                        best["crop"], best["kp_crop"], best["good_matches"]
                    ))
                    wrote_det = wrote_match = True

            elif tracker_initialized and frame_count % 10 == 0:
                if last_tracked_crop is not None and last_tracked_crop.size > 0:
                    kp_c, des_c = sift.detectAndCompute(
                        cv2.cvtColor(last_tracked_crop, cv2.COLOR_BGR2GRAY), None
                    )
                    if des_c is not None and len(des_c) >= 2:
                        matches = flann.knnMatch(draw_des_target, des_c, k=2)
                        good    = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

                        if len(good) >= MIN_MATCH_COUNT:
                            x1, y1, x2, y2 = last_tracked_box.astype(int)
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                            out_det.write(annotate(frame, last_tracked_box, "#00AAFF"))
                            out_match.write(build_match_frame(
                                draw_target_color, draw_kp_target,
                                last_tracked_crop, kp_c, good
                            ))
                            wrote_det = wrote_match = True

            if not wrote_det and tracker_initialized:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h     = [int(v) for v in bbox]
                    last_tracked_box  = np.array([x, y, x + w, y + h], dtype=float)
                    last_tracked_crop = frame[y:y + h, x:x + w]

                    out_det.write(annotate(frame, last_tracked_box, "#00FF00"))
                    wrote_det = True
                else:
                    tracker_initialized = False

            if not wrote_det:
                out_det.write(frame)
            if not wrote_match:
                out_match.write(blank_match_frame())

            if frame_count % 30 == 0 and last_tracked_crop is not None \
                    and last_tracked_crop.size > 0:
                upd_gray = cv2.cvtColor(last_tracked_crop, cv2.COLOR_BGR2GRAY)
                kp_u, des_u = sift.detectAndCompute(upd_gray, None)
                if des_u is not None and len(des_u) >= MIN_MATCH_COUNT:
                   
                    draw_des_target = des_u
                    draw_kp_target  = kp_u
                    draw_target_color = last_tracked_crop.copy()
                    print(f"[Target Updated] frame {frame_count}")

        except Exception as e:
            print(f"[Frame {frame_count} error] {e}")
            out_det.write(frame)
            out_match.write(blank_match_frame())

finally:
    cap.release()
    out_det.release()
    out_match.release()
    print("Done →", output_video_path)
