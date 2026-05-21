



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

## 📊 3. Visual Results & Showcase (Արդյունքներ)

Here is how the hybrid pipeline performs under strict verification and dynamic tracking constraints.

### A. Real-Time Video Tracking & Target Lock-On
*The baseline target is verified via SIFT geometry on the GPU, after which the lightweight CSRT tracker maintains a high frame-rate lock-on.*

| 🎯 Reference Target Image | 🔒 Target Verification & Active Tracking |
|---|---|
| <img src="https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f" width="220" alt="Target Reference"> <br> *Figure 1: The specific structural target used for matching.* | <video src="https://github.com/user-attachments/assets/ba271dc7-2e47-4c93-9a3b-13f34fed960c" width="100%" autoplay loop muted controls></video> <br> *Figure 2: Grounding DINO + SIFT locking on and offloading to CSRT.* |

### B. Image Collection Batch Verification (Scale & Rotation Invariance)
*When evaluating static datasets or large map grids, the system maps precise point-to-point visual correlation lines to guarantee absolute structural alignment.*

| 🖼️ Multi-Object Scene Detection | 📐 SIFT Point Correspondence Mapping |
|---|---|
| <img src="https://github.com/user-attachments/assets/df0f3451-1c96-47d0-9aad-c736274e02c1" width="180" alt="Target Object"> <br> *Figure 3: Candidate region proposal.* | <video src="https://github.com/user-attachments/assets/67189a73-65f5-4f60-900c-e2a40df3c9d4" width="100%" autoplay loop muted controls></video> <br> *Figure 4: Invariant feature matching overlay.* |

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







