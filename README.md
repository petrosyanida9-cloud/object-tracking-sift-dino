Hybrid Zero-Shot Object Detection, Keypoint Verification & Tracking

A high-accuracy computer vision pipeline combining GPU-accelerated zero-shot detection (Grounding DINO), invariant feature verification (SIFT + FLANN), and high-speed state tracking (CSRT) — engineered for zero false-positive scenarios such as satellite imagery analysis and tactical target tracking.


📌 The Problem
Modern CV systems break down in real-world tactical or satellite scenarios due to four core bottlenecks:
#BottleneckWhy It Fails1Heavy InferenceRunning Grounding DINO on every frame is prohibitively slow at high resolution2Scale & Rotation VarianceDrone/satellite objects constantly shift orientation (0–360°) and scale — pyramids are memory-hungry3Illumination InstabilitySun glare, shadows, weather, and day/night cycles defeat pixel-level template matching4Tracker DriftLightweight trackers (CSRT) have no semantic recovery — once lost, the target is gone

💡 Our Hybrid Solution
The pipeline cleanly separates concerns into three specialized, cooperating layers:
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
Why this works:

⚡ Speed — Heavy GPU inference runs only when needed; CSRT handles the in-between frames
📐 Invariance — SIFT is mathematically resistant to scale and rotation by design — no brute-force image pyramids
🌦️ Illumination Robustness — Grounding DINO's semantic understanding doesn't rely on pixel-level appearance
🔄 Self-Recovery — Drift is detected automatically, triggering a full re-localization cycle


📊 Results & Demos
🎥 Part 1 — Real-Time Video Tracking & Target Lock-On
The pipeline locks onto a verified target and hands off to CSRT for smooth, low-latency frame tracking. If the tracker drifts, Grounding DINO + SIFT automatically re-engage.
<table>
  <tr>
    <td align="center" width="240">
      <b>Reference Target</b><br>
      <!-- PLACEHOLDER: Replace src with your uploaded target.jpg GitHub asset URL -->
      <!-- Example: https://github.com/user-attachments/assets/YOUR-TARGET-IMAGE-ID -->
      <img src="https://github.com/user-attachments/assets/d7175ae2-9654-4cb8-b92a-98355ff8574f" width="220" alt="Reference target used for SIFT lock-on"><br>
      <i>The reference image fed to SIFT for geometric verification</i>
    </td>
    <td align="center">
      <b>Live Tracking Demo</b><br>
      <!-- 
        HOW TO EMBED YOUR VIDEO:
        1. Go to any GitHub Issue in your repo (or create a new one — you don't need to submit it)
        2. Drag & drop output_detected_video.mp4 into the comment box
        3. Wait for it to upload — GitHub gives you a URL like:
           https://github.com/user-attachments/assets/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        4. Replace the src below with that URL
      -->
      <video src="https://github.com/user-attachments/assets/ba271dc7-2e47-4c93-9a3b-13f34fed960c" width="480" controls autoplay loop muted></video><br>
      <i>Grounding DINO + SIFT lock-on, then CSRT takeover</i>
    </td>
  </tr>
</table>

📐 Part 2 — Batch Image Verification (Scale & Rotation Invariance)
For static datasets and large map grids, the system draws precise point-to-point SIFT correspondence lines to confirm structural alignment — no false positives.
<table>
  <tr>
    <td align="center" width="240">
      <b>Example: Matched Candidate</b><br>
      <!-- 
        PLACEHOLDER — YOUR SIFT MATCH IMAGE 1
        Upload your SIFT match result image to a GitHub Issue (same drag & drop method)
        then replace this src with the asset URL.
        Suggested: one of your matching_visuals_0.75/ output frames
      -->
      <img src="YOUR_SIFT_MATCH_IMAGE_1_URL_HERE" width="220" alt="SIFT match example 1"><br>
      <i>True positive: correspondence lines confirm structural match</i>
    </td>
    <td align="center" width="240">
      <b>Example: Rejected Candidate</b><br>
      <!-- 
        PLACEHOLDER — YOUR SIFT MISMATCH / REJECTION IMAGE
        Upload a frame showing a rejected candidate (too few matches)
      -->
      <img src="YOUR_SIFT_REJECTION_IMAGE_URL_HERE" width="220" alt="SIFT rejection example"><br>
      <i>False positive eliminated: insufficient geometric matches</i>
    </td>
    <td align="center">
      <b>SIFT Correspondence Mapping Demo</b><br>
      <!-- 
        HOW TO EMBED YOUR VIDEO (output_final(3).mp4):
        Same process — drag & drop into a GitHub Issue comment, copy the asset URL
        Replace the src below with that URL
      -->
      <video src="https://github.com/user-attachments/assets/67189a73-65f5-4f60-900c-e2a40df3c9d4" width="420" controls autoplay loop muted></video><br>
      <i>Live SIFT keypoint correspondence overlay across batch images</i>
    </td>
  </tr>
</table>

💡 Uploading images: Same method as video — drag & drop into a GitHub Issue comment box, copy the generated https://github.com/user-attachments/assets/... URL, and paste it into the src above.


🛠️ Repository Structure
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

⚙️ Technical Parameters
ParameterValueRationaleSIFT Lowe's Ratio0.75Balances match precision vs. recallMinimum SIFT Matches≥ 7Prevents geometric false positivesTrackerCSRTBest accuracy among OpenCV lightweight trackersDetection BackendGrounding DINOZero-shot, text-promptable, illumination-robustAccelerationPyTorch CUDAGPU inference for DINO proposals

🚀 Setup
bashpip install -r requirements.txt
Real-time video tracking:
bashpython src/video_pipeline.py --target data/inputs/target.jpg --video data/inputs/video.mp4
Batch image verification:
bashpython src/image_pipeline.py --target data/inputs/target.jpg --images data/inputs/images/

📐 How the Math Works
SIFT Scale-Space Extrema Detection:
L(x,y,σ)=G(x,y,σ)∗I(x,y)L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)L(x,y,σ)=G(x,y,σ)∗I(x,y)
Keypoints are localized at scale-space extrema of the Difference-of-Gaussians, giving inherent scale invariance.
Lowe's Ratio Test (false-match rejection):
d1d2<0.75\frac{d_1}{d_2} < 0.75d2​d1​​<0.75
Only matches where the nearest neighbor is significantly closer than the second-nearest are accepted, eliminating ambiguous correspondences.
FLANN Approximate Nearest Neighbor reduces match search from O(n2)O(n^2)
O(n2) brute-force to O(nlog⁡n)O(n \log n)
O(nlogn), enabling real-time performance on large keypoint sets.

📋 Requirements
See requirements.txt — key dependencies:

torch + CUDA
groundingdino
opencv-python
numpy
