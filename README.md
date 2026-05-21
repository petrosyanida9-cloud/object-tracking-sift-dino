# Hybrid Zero-Shot Object Detection, Keypoint Verification & Tracking

> A high-accuracy computer vision pipeline combining GPU-accelerated zero-shot detection (Grounding DINO), invariant feature verification (SIFT + FLANN), and high-speed state tracking (CSRT) — built for zero false-positive scenarios such as satellite imagery analysis and tactical target tracking.

---

## 📌 The Problem

| # | Bottleneck | Why It Fails |
|---|-----------|-------------|
| 1 | **Heavy Inference** | Running Grounding DINO on every frame is prohibitively slow at high resolution |
| 2 | **Scale & Rotation Variance** | Drone/satellite objects constantly shift orientation (0–360°) and scale |
| 3 | **Illumination Instability** | Sun glare, shadows, and day/night cycles defeat pixel-level template matching |
| 4 | **Tracker Drift** | Lightweight trackers (CSRT) have no semantic recovery — once lost, the target is gone |

---

## 💡 Hybrid Solution
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — GPU: Grounding DINO (Zero-Shot Semantic Proposals)   │
│           Text prompt → candidate bounding boxes                │
└────────────────────────┬────────────────────────────────────────┘
│ candidates
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — GPU: SIFT + FLANN (Invariant Feature Verification)   │
│           Cross-verify vs target.jpg  │  Lowe's Ratio = 0.75    │
│           Scale & rotation invariant  │  Min matches ≥ 7        │
└────────────────────────┬────────────────────────────────────────┘
│ verified lock
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — CPU: OpenCV CSRT Tracker (High-Speed Frame Tracking) │
│           Neural net sleeps  │  Smooth real-time performance    │
│           Auto wake on drift │  Re-triggers Layers 1 & 2        │
└─────────────────────────────────────────────────────────────────┘
- **⚡ Speed** — GPU inference runs only when needed; CSRT handles in-between frames
- **📐 Invariance** — SIFT is mathematically resistant to scale and rotation by design
- **🌦️ Illumination Robustness** — Grounding DINO's semantic understanding ignores pixel-level changes
- **🔄 Self-Recovery** — Drift triggers automatic re-localization via DINO + SIFT

---

## 📊 Results & Demos

### 🎥 Part 1 — Real-Time Video Tracking & Target Lock-On

| Reference Target | Demo |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f" width="240"> | [![Demo](https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f)](REPLACE_WITH_res_vd_URL) |
| *Target reference used for SIFT lock-on* | *▶ Click to watch — DINO + SIFT lock-on → CSRT takeover* |

### 📐 Part 2 — Batch Image Verification (Scale & Rotation Invariance)

| Reference Target | Demo |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/df0f3451-1c96-47d0-9aad-c736274e02c1" width="240"> | [![Demo](https://github.com/user-attachments/assets/df0f3451-1c96-47d0-9aad-c736274e02c1)](REPLACE_WITH_res_vd1_URL) |
| *Candidate region from batch image* | *▶ Click to watch — SIFT correspondence lines overlay* |

---

## 🛠️ Repository Structure
object-tracking-sift-dino/
├── data/
│   ├── inputs/
│   │   ├── target.jpg                   # Reference image for SIFT verification
│   │   ├── video.mp4                    # Input video for real-time tracking
│   │   └── images/                      # Static images for batch processing
│   └── outputs/
│       ├── output_detected_video.mp4    # Annotated tracking output video
│       ├── output_accurate/             # DINO-verified bounding box frames
│       └── matching_visuals_0.75/       # SIFT correspondence visualizations
├── src/
│   ├── video_pipeline.py                # Real-time: DINO + SIFT + CSRT tracker
│   └── image_pipeline.py               # Batch: DINO + SIFT + FLANN verification
├── requirements.txt
└── README.md
---

## ⚙️ Technical Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SIFT Lowe's Ratio | `0.75` | Balances match precision vs. recall |
| Minimum SIFT Matches | `≥ 7` | Prevents geometric false positives |
| Tracker | `CSRT` | Best accuracy among OpenCV lightweight trackers |
| Detection Backend | `Grounding DINO` | Zero-shot, text-promptable, illumination-robust |
| Acceleration | `PyTorch CUDA` | GPU inference for DINO proposals |

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

**Real-time video tracking:**
```bash
python src/video_pipeline.py --target data/inputs/target.jpg --video data/inputs/video.mp4
```

**Batch image verification:**
```bash
python src/image_pipeline.py --target data/inputs/target.jpg --images data/inputs/images/
```

---

## 📐 How the Math Works

**SIFT Scale-Space Extrema Detection:**

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

Keypoints are detected at scale-space extrema of the Difference-of-Gaussians, giving inherent scale invariance.

**Lowe's Ratio Test:**

$$\frac{d_1}{d_2} < 0.75$$

Only matches where the nearest neighbor is significantly closer than the second-nearest survive, eliminating ambiguous correspondences.

**FLANN** reduces nearest-neighbor search from $O(n^2)$ to $O(n \log n)$, enabling real-time performance on large keypoint sets.

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) — key dependencies: `torch` (CUDA), `groundingdino`, `opencv-python`, `numpy`
