Markdown# Hybrid Object Detection & Tracking (Grounding DINO + SIFT)

An intelligent computer vision repository designed to lock onto, track, and verify specific visual targets within dynamic video feeds and static image collections. By combining zero-shot deep learning detection with classic high-accuracy keypoint matching, this pipeline offers robust re-localization and precise verification.

## 🚀 Repository Structure

```text
object-tracking-sift-dino/
├── data/
│   ├── inputs/
│   │   ├── target.jpg          # Target object to search for
│   │   ├── video.mp4           # Input video for tracking
│   │   └── images/             # Input folder for static image testing
│   └── outputs/
│       ├── output_detected_video.mp4
│       ├── output_accurate/    # DINO + SIFT verified images
│       └── matching_visuals_0.75/
├── src/
│   ├── video_pipeline.py       # Hybrid Video Tracking (DINO + SIFT + CSRT)
│   └── image_pipeline.py       # Image Collection Verification (DINO + SIFT + FLANN)
├── .gitignore
├── README.md
└── requirements.txt

## 🛠️ System Architectures

### 1. Video Pipeline (`src/video_pipeline.py`)
- **Initialization:** Grounding DINO scans frames with text prompts to locate potential targets.
- **Verification:** Candidates are cross-verified against `target.jpg` using SIFT feature matching with Lowe's Ratio test (0.75) to guarantee a minimum match count (>= 7).
- **Tracking:** Once a lock-on is achieved, an efficient **OpenCV CSRT tracker** takes over to handle frame-by-frame updates without running heavy neural network inferences constantly.
- **Recovery:** If the tracker loses the target, Grounding DINO re-triggers automatically to re-localize the object.

### 2. Image Collection Pipeline (`src/image_pipeline.py`)
- Processes batches of static images inside `data/inputs/images/`.
- Extracts object crops using Grounding DINO based on custom semantic text prompts.
- Runs **SIFT + FLANN Based Matcher** on bounding box crops to find highly accurate feature correspondences, outputting clear visual correlation lines for targets matching the threshold criteria.

## 📦 Installation & Usage

Ensure you have a Linux environment with Python 3.10+ and the required packages installed:

```bash
# Clone the repository
git clone [https://github.com/petrosyanida9-cloud/object-tracking-sift-dino.git](https://github.com/petrosyanida9-cloud/object-tracking-sift-dino.git)
cd object-tracking-sift-dino

# Install dependencies
pip install -r requirements.txt

Running the pipelines:

Place your sample assets inside data/inputs/ and run either script:
Bash

python src/video_pipeline.py
python src/image_pipeline.py


