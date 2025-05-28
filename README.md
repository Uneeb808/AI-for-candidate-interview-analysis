# InsightHire AI

A next-generation web-based interview analysis platform that brings objectivity, consistency and deep behavioral insights to remote interviews.

---

## 📋 Problem Statement

In today’s remote-first world, traditional interviews rely heavily on subjective human judgment:

- **Bias & Inconsistency:** Different interviewers evaluate the same candidate differently.  
- **Missed Non-verbal Cues:** Stress, confidence, engagement and other signals are hard to gauge over video.  
- **Limited Data:** Interviewers can’t easily quantify performance trends or compare candidates objectively.  
- **Remote Challenges:** In digital interactions, emotional and behavioral cues are even harder to assess.

**Who is affected?**  
- **Recruiters & Hiring Managers:** Struggle to consistently rank candidates.  
- **Candidates:** Receive little actionable feedback.  
- **Educators & Therapists:** Can’t scale remote engagement or stress monitoring.

---

## 🛠 Our Solution

**InsightHire AI** integrates real-time computer vision, audio analytics, and ML to compute multidimensional behavioral scores, displayed in an interactive dashboard:

1. **Emotion Analysis** via facial-expression recognition + voice pitch & energy metrics.  
2. **Eye Contact Tracking** using MediaPipe FaceMesh.  
3. **Posture & Movement** scoring via MediaPipe Pose.  
4. **Anxiety Detection** through blink count, face touches, jaw tension.  
5. **Expressive Speech**—pitch variability, range, energy variance.  
6. **Composite “Engagement”, “Confidence”, “Stress” & “Professionalism”** time-series data and summary averages.  

At the end, an **Overall Score** (0–10) combines all four pillars with customizable weights.

---

## 🔍 Parameter Definitions & Scoring

| **Parameter**        | **How We Measure**                                                            | **Score Calculation**                                                                                           |
|----------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Emotion**          | 1. Facial expressions (HSEmotionRecognizer)<br>2. Voice pitch & energy metrics  | 1. **Audio Emotion Value** (0–1) via prosody/sentiment model<br>2. **Facial Emotion Proxy** (0–1) via landmark averages<br>3. **Pitch Variability** (std Hz → normalized 0–1)<br>4. **Pitch Range** (Hz range → norm 0–1)<br>5. **Energy Variance** (RMS coeff. of var. → norm 0–1)<br><br>**Weighted base:** 35% audio + 35% facial + 15% pitch + 15% energy<br>**Adjustments:** penalties for mixed/unnatural signals, bonuses for consistency<br>→ **Scaled to 0–10** |
| **Eye Contact**      | Gaze offset vs. face-center + EAR threshold for open eyes                      | - Compute average EAR (eye aspect ratio) to ensure eyes open<br>- Measure gaze-center offset normalized to frame width<br>- **Score/frame:** `10 * (1 – offset/threshold)` clamped [0–10]<br>- **Final:** `10 * (1 – poor_frames/total_frames)`                      |
| **Posture**          | Shoulder & hip tilt stability                                                   | - Measure frame-to-frame shoulder & hip height difference<br>- Detect shifts > 0.02 normalized units<br>- **Frame shifts ratio** → penalize: `score = max(0, 10 – shift_ratio*15)` |
| **Anxiety**          | Blink rate, face touches, jaw tension, eye avoidance                            | - Count blinks (EAR drops)<br>- Detect face touches (finger ↔ nose < 0.05 norm)<br>- Jaw width < threshold → tension<br>- Eye avoidance frames<br>- **Combined ratio:** `(blinks + touches + tension_events + avoidance)/frames` → `anxiety_score = 10*(1 – ratio)` |

### Composite “New” Categories (per-frame → summary)

| Category         | Formula (per frame)                                                                                                 |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| **Engagement**   | 0.6×(eye_contact_frame) + 0.4×(posture_frame)                                                                        |
| **Confidence**   | 0.5×(eye_contact_frame) + 0.4×(pitch_range*10) + 0.1×(posture_frame)                                                 |
| **Stress**       | `10 – (0.6*(10 – anxiety_frame) + 0.4*facial_emotion_frame)`                                                         |
| **Professionalism** | 0.5×(posture_frame) + 0.5×(energy_variance*10)                                                                   |

Each is aggregated across all frames to yield a **final average (0–10)**.

---

## 📊 Final Score & Visualization

- **Weights:** Emotion 40% + Eye Contact 20% + Posture 20% + Anxiety 20%  
- **Interactive Timeline:** Chart.js line graph of Engagement, Confidence, Stress, Professionalism over time.  
- **Dashboard:** Circular progress rings for each pillar, detailed counts, head-nod tally, pitch & energy stats.

---

## File Structure & Descriptions

- **app.py**  
  This is the main backend file built with Flask. It handles:
  - Accepting video uploads.
  - Video processing and audio extraction using **OpenCV**, **MediaPipe**, and **Librosa**.
  - Running the core machine learning models for emotion recognition and metric computation.
  - Compiling the analysis results and rendering appropriate HTML pages (e.g., result.html).

- **index.html**  
  The landing page of the application where users:
  - Record their interview sessions with their webcam.
  - Use interactive controls (implemented with **JavaScript**) to start and stop recordings.
  - Experience a sleek, modern design thanks to advanced **CSS** styling.

- **result.html**  
  The results page displays the detailed analysis of the interview session. It includes:
  - Comprehensive metrics such as overall score, emotion score, posture, and stress indicators.
  - A dynamic time-series graph rendered using **Chart.js** (integrated via **JavaScript**) to visualize performance over time.
  - A clean, card-based layout styled with modern **CSS** that ensures a professional look and responsive design.

- **error.html**  
  This page is used to display custom error messages in a user-friendly manner. It maintains consistent aesthetics with the rest of the application and helps guide the user when issues arise.

## Installation Required-

1. **Prerequisites:**  
Make sure to install the following Python libraries before running the project:

```bash
pip install flask
pip install opencv-python
pip install mediapipe
pip install hsemotion-onnx
pip install librosa
pip install matplotlib
pip install numpy
pip install Pillow
pip install ffmpeg
```

## 🚀 How to Run the Application

### 1. Navigate to the Project Directory in your terminal

```bash
cd your_project_folder

python app.py


