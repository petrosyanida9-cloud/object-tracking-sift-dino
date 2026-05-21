



# Hybrid Zero-Shot Object Detection, Keypoint Verification & Tracking

An advanced computer vision pipeline that combines deep learning zero-shot object detection with classic high-precision feature matching and high-speed state tracking. This hybrid architecture is specifically optimized for high-accuracy targets where false positives must be zero, such as locating specific structures in satellite/map imagery or maintaining a continuous lock on specific moving targets.

---

## 📌 1. The Problem Statement 
Modern computer vision systems face critical failures when deployed in real-world tactical or satellite tracking scenarios due to four main bottlenecks:
1. **The Heavy Inference Bottleneck:** Running large transformer-based object detectors (like Grounding DINO) on every single frame of a high-resolution video stream is computationally prohibitive and causes massive latency.
2. **Extreme Scale & Rotation Variations:** Objects captured from drones or satellites constantly change orientation ($360^\circ$ rotation) and altitude (drastic scale differences). Naive tracking or brute-force image pyramids fail or consume too much memory.
3. **Illumination Inconstancy:** Sun glare, moving shadows, weather conditions, and day/night shifts radically change the pixel-level appearance of the target, tripping up basic template-matching algorithms.
4. **The Tracker Drift Dilemma:** Lightweight visual trackers (like CSRT) are fast but lack semantic intelligence. Once they lose the target due to rapid occlusion or background noise, they cannot self-verify or recover.

---

## 💡 2. Our Hybrid Solution 
This repository presents an elegant, mathematically robust pipeline that achieves **blazing-fast speed** and **multi-environmental invariance** by separating tasks into specialized layers:

* **⚡ GPU-Accelerated Zero-Shot Proposals:** We leverage **Grounding DINO** combined with **PyTorch CUDA acceleration** to parse complex scenes instantly based on text prompts. This provides semantic-level candidate localization under extreme illumination shifts without custom training.
* **📐 Invariant Feature Verification (Scale & Rotation):** To eliminate false positives, candidates are cross-verified against `target.jpg`. By using **SIFT + FLANN**, we exploit SIFT’s intrinsic mathematical resistance to scale and rotation. This avoids inefficient brute-force image pyramids while maintaining strict geometric verification ($\text{Lowe's Ratio} = 0.75$, $\text{Min Matches} \ge 7$).
* **🏎️ Offloaded CPU Tracking:** Once a high-confidence lock-on is verified via GPU, the heavy neural network goes to sleep. A fast, highly efficient **OpenCV CSRT Tracker** takes over frame-by-frame monitoring on the CPU, achieving smooth real-time performance.
* **🔄 Automatic Drift Recovery:** If the tracker encounters severe obstruction and drifts, the pipeline automatically wakes up the GPU-bound Grounding DINO + SIFT loop to re-localize and re-lock the target.

---

## 📊 3. Visual Results & Showcase 

Here is how the pipeline performs under strict verification constraints. 

### A. Real-Time Video Tracking & Lock-On
*Below is the automated process: Grounding DINO fires on the GPU ➡️ SIFT verifies the target geometry ➡️ CSRT locks onto the tracking states.*

| 🔒 Target Verification & Lock-On | 🏎️ High-Speed Active Tracking |
|---|---|
| <img src="data/outputs/output_accurate/det_frame_sample.jpg" width="100%" alt="Target Locked via DINO+SIFT"> <br> *Figure 1: Initial detection and validation using CUDA & strict SIFT matching.* | <img src="data/outputs/output_accurate/tracking_sample.gif" width="100%" alt="Active CSRT Tracking Mode"> <br> *Figure 2: Lightweight tracking loop running at high frame rates.* |

### B. Image Collection Batch Verification (Scale & Rotation Invariance)
*For individual image analysis or satellite grids, the pipeline isolates candidates and maps point-to-point visual correlation lines to prove absolute structural alignment, regardless of orientation.*

<p align="center">
  <img src="data/outputs/matching_visuals_0.75/match_sample.jpg" width="90%" alt="SIFT Point-to-Point Correspondence">
  <br>
  <i>Figure 3: Detailed SIFT + FLANN verification lines showing robust invariant matching on target crops.</i>
</p>

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


https://github.com/user-attachments/assets/7c120bc5-9f60-4414-afba-359da6bb4d51


https://github.com/user-attachments/assets/4a6a8c63-ecfa-4d84-b77c-b13b00de24a5
<img width="292" height="213" alt="target" src="https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f" />

<img width="70" height="101" alt="target_vd" src="https://github.com/user-attachments/assets/df0f3451-1c96-47d0-9aad-c736274e02c1" />







