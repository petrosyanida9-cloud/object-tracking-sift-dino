# Hybrid Zero-Shot Object Detection, Keypoint Verification & Tracking

<div align="center">

### ⚡ Grounding DINO + SIFT + FLANN + CSRT Hybrid Vision Pipeline

An advanced hybrid computer vision architecture optimized for:

**Satellite Intelligence • UAV Tracking • Structural Verification • Real-Time Target Lock-On**

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-ZeroShot-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

# 🚀 Overview

This repository presents a **high-performance hybrid computer vision pipeline** that combines:

- 🧠 **Zero-shot semantic object detection**
- 📐 **Invariant geometric verification**
- 🏎️ **Ultra-fast visual tracking**

The system is engineered for **extreme precision environments where false positives must approach zero**.

Typical applications include:

- 🛰️ Satellite imagery analysis
- 🚁 Drone/UAV surveillance
- 🏗️ Structural localization
- 🎯 Persistent target lock-on
- 🗺️ Map-grid infrastructure verification
- 🔍 Rotation-invariant object search

---

# ⚠️ The Core Problem

Traditional object detection pipelines fail in real-world tactical scenarios due to several major bottlenecks.

---

## 1️⃣ Heavy Inference Bottleneck

Large transformer detectors such as **Grounding DINO** are computationally expensive when executed on every frame.

This causes:

- GPU overload
- High latency
- Poor real-time performance

---

## 2️⃣ Extreme Scale & Rotation Variations

Targets captured from drones or satellites continuously change:

- scale
- altitude
- orientation
- viewing angle

Naive template matching becomes unstable and inefficient.

---

## 3️⃣ Illumination Inconstancy

Environmental conditions drastically alter object appearance:

- shadows
- glare
- weather
- day/night shifts
- seasonal variation

Classical pixel-based matching fails under these conditions.

---

## 4️⃣ Tracker Drift

Fast trackers such as CSRT are lightweight but semantically blind.

Once drift occurs:
- the tracker cannot self-verify
- false lock-ons appear
- recovery becomes unreliable

---

# 💡 Hybrid Solution Architecture

This project separates tasks into specialized computational layers.

---

## ⚡ Stage 1 — Zero-Shot Proposal Generation

Using:

- Grounding DINO
- PyTorch CUDA acceleration

the system generates semantic candidate regions using text prompts.

Advantages:
- illumination robustness
- generalized detection
- no custom training required

---

## 📐 Stage 2 — Geometric Verification

Every candidate undergoes strict verification against `target.jpg`.

Technologies:
- SIFT keypoints
- FLANN matching

SIFT provides intrinsic robustness to:

- scale variation
- 360° rotation
- moderate affine distortion

Verification Constraints:

```math
Lowe's\ Ratio = 0.75
```

```math
Minimum\ Good\ Matches \ge 7
```

This stage removes false positives through geometric consistency.

---

## 🏎️ Stage 3 — High-Speed Tracking

Once verified:

- the neural detector sleeps
- GPU usage drops dramatically
- CSRT tracker takes over

Result:
- real-time performance
- smooth lock-on
- lightweight monitoring

---

## 🔄 Stage 4 — Automatic Recovery

If tracker drift occurs:

1. CSRT failure is detected
2. Grounding DINO reactivates
3. SIFT verification reruns
4. target is reacquired automatically

---

# 🧩 Full Pipeline Flow

```text
Input Frame
     ↓
Grounding DINO Detection (GPU)
     ↓
Candidate Region Proposals
     ↓
SIFT + FLANN Verification
     ↓
Verified Target Lock
     ↓
CSRT Tracker Activation (CPU)
     ↓
Real-Time Tracking
     ↓
Tracker Drift?
 ├── No → Continue Tracking
 └── Yes → Re-run DINO + SIFT
```

---

# 📊 Visual Showcase

---

# 🎯 Reference Target

<div align="center">

<img src="https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f" width="320"/>

<br>

<i>Figure 1 — Structural reference target used for invariant verification.</i>

</div>

---

# 🎥 Real-Time Tracking Demo

https://github.com/user-attachments/assets/ba271dc7-2e47-4c93-9a3b-13f34fed960c

<div align="center">

<i>Figure 2 — Grounding DINO + SIFT verification followed by CSRT real-time tracking.</i>

</div>

---

# 📐 Geometric Verification Demo

<div align="center">

<img src="https://github.com/user-attachments/assets/df0f3451-1c96-47d0-9aad-c736274e02c1" width="260"/>

<br>

<i>Figure 3 — Candidate structural proposal extracted from dataset images.</i>

</div>

---

# 🔍 SIFT Correspondence Visualization

https://github.com/user-attachments/assets/67189a73-65f5-4f60-900c-e2a40df3c9d4

<div align="center">

<i>Figure 4 — Invariant feature correspondence and geometric alignment verification.</i>

</div>

---

# 🛠️ Repository Structure

```text
object-tracking-sift-dino/
├── data/
│   ├── inputs/
│   │   ├── target.jpg
│   │   ├── video.mp4
│   │   └── images/
│   │
│   └── outputs/
│       ├── output_detected_video.mp4
│       ├── output_accurate/
│       └── matching_visuals_0.75/
│
├── src/
│   ├── video_pipeline.py
│   └── image_pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📂 Components

| File | Description |
|---|---|
| `video_pipeline.py` | Real-time hybrid tracking pipeline |
| `image_pipeline.py` | Batch image verification |
| `target.jpg` | Reference object |
| `video.mp4` | Input tracking video |
| `output_detected_video.mp4` | Generated output video |
| `matching_visuals_0.75/` | SIFT correspondence visualizations |

---

# ⚙️ Installation

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/object-tracking-sift-dino.git
cd object-tracking-sift-dino
```

---

## 2️⃣ Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

---

# 🎥 Real-Time Video Tracking

```bash
python src/video_pipeline.py
```

Outputs:
- verified detections
- smooth tracking
- annotated video generation

---

# 🖼️ Batch Image Verification

```bash
python src/image_pipeline.py
```

Outputs:
- candidate detections
- SIFT verification overlays
- feature correspondence maps

---

# 🔬 Technology Stack

| Component | Technology |
|---|---|
| Detection | Grounding DINO |
| Deep Learning | PyTorch |
| GPU Acceleration | CUDA |
| Feature Extraction | SIFT |
| Feature Matching | FLANN |
| Tracking | OpenCV CSRT |
| Language | Python |

---

# 📈 Advantages

| Traditional Pipeline | Hybrid Pipeline |
|---|---|
| Heavy inference every frame | Detector sleeps after lock |
| Weak rotation handling | Full SIFT invariance |
| High GPU load | CPU tracker offloading |
| Tracker drift unrecoverable | Automatic re-localization |
| Higher false positives | Strict geometric verification |

---

# 🎯 Applications

- 🛰️ Satellite intelligence
- 🚁 UAV target tracking
- 🏗️ Structural verification
- 🗺️ Geospatial analysis
- 🎯 Persistent surveillance
- 🔍 Rotation-invariant localization
- ⚡ High-precision tactical vision systems

---

# 🔮 Future Improvements

- Multi-object tracking
- Kalman filtering
- Optical flow fusion
- DeepSORT integration
- TensorRT acceleration
- Distributed multi-camera support
- ORB lightweight embedded mode

---

# 👨‍💻 Author

Developed as a high-precision research-oriented hybrid computer vision system combining:

- semantic intelligence
- geometric invariance
- real-time performance

into a unified architecture.

---

# 📜 License

MIT License

---

# ⭐ Support The Project

If this repository helped you:

⭐ Star the repository  
🍴 Fork the project  
🧠 Contribute improvements  

---

<div align="center">

### ⚡ Hybrid Vision Intelligence Pipeline

Built for precision-critical computer vision systems.

</div>
