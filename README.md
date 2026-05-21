# Hybrid Zero-Shot Object Detection, Keypoint Verification & Tracking

An advanced computer vision pipeline that combines deep learning zero-shot object detection with classic high-precision feature matching and high-speed state tracking. This hybrid architecture is specifically optimized for high-accuracy targets where false positives must be zero, such as locating specific structures in satellite/map imagery or maintaining a continuous lock on specific moving targets.

---

## 📌 1. The Problem Statement
Modern computer vision systems often struggle with two fundamental challenges when deployed in precision-critical tracking scenarios:
1. **The Heavy Inference Bottleneck:** Running large transformer-based object detectors (like Grounding DINO) on every single frame of a high-resolution video stream is computationally prohibitive and causes latency.
2. **The Tracker Drift & False Positive Dilemma:** Standard lightweight visual trackers (like CSRT or KCF) are fast but lack semantic intelligence. They easily lose the target (drift) during fast motion, occlusion, or background noise, and they cannot self-verify if they are tracking the correct asset or a background object.

---

## 💡 2. Our Hybrid Solution
This repository presents an elegant split-architecture pipeline that solves both constraints by separating **Localization**, **Strict Verification**, and **High-Speed Tracking**:

* **Zero-Shot Candidate Proposal:** We leverage **Grounding DINO** using semantic text prompts (e.g., `small object . tiny structure . central building .`) to locate target candidates without training a custom object detector.
* **Rigorous SIFT Verification:** To eliminate false detections, candidates are validated against a local reference image (`target.jpg`). Feature extraction via **SIFT** and correspondence matching via **FLANN** filter out noise using a tight **Lowe's Ratio (0.75)**. A target lock-on is granted *only* if geometric consistency meets a minimum threshold of **>= 7 strict keypoint matches**.
* **Offloaded Continuous Tracking:** Once a target is verified, the heavy neural network inference goes to sleep. A fast, CPU-efficient **OpenCV CSRT Tracker** takes over to maintain real-time tracking across video frames.
* **Automated Drift Recovery:** If the CSRT tracker's confidence drops or loses the target, the pipeline automatically re-triggers the Grounding DINO + SIFT loop to re-localize and re-lock the asset.

---

## 📊 3. Visual Results & Outputs
Here is the pipeline executing successfully under strict constraints.

### A. Real-Time Video Tracking & Lock-On
*Below you can see Grounding DINO initially detecting the target, SIFT verifying the keypoint geometry, and the CSRT tracker taking over with high frame rates.*

| Initial Target Lock & Verification | Continuous CSRT Tracker States |
|---|---|
| <!-- Replace 'data/outputs/verification.png' with your image or GIF path --> <img src="data/outputs/output_accurate/det_frame_sample.jpg" width="100%" alt="Target Locked"> | <!-- Replace with another screenshot or video link --> <img src="data/outputs/output_accurate/tracking_sample.gif" width="100%" alt="Tracking Mode"> |

### B. Image Collection Batch Verification (SIFT Feature Matching Lines)
*For individual image analysis, the pipeline isolates candidate bounding boxes and maps point-to-point correspondence to prove mathematical alignment.*

<!-- Place one of your best matching output images here -->
![SIFT Mapping Lines](data/outputs/matching_visuals_0.75/match_sample.jpg)

---

## 🛠️ 4. Repository Architecture

```text
object-tracking-sift-dino/
├── data/
│   ├── inputs/
│   │   ├── target.jpg          # Reference target image to find/verify
│   │   ├── video.mp4           # Input video stream for real-time tracking
│   │   └── images/             # Directory containing static images for batch testing
│   └── outputs/
│       ├── output_detected_video.mp4  # Generated high-speed tracking video
│       ├── output_accurate/           # Bounding box annotated frames (DINO verified)
│       └── matching_visuals_0.75/     # SIFT verification line plots
├── src/
│   ├── video_pipeline.py       # Live Tracking Script (DINO + SIFT + CSRT Tracker)
│   └── image_pipeline.py       # Static Batch Verification Script (DINO + SIFT + FLANN)
├── .gitignore
├── README.md
└── requirements.txt
