"""
Full Updated app.py for Interview Analysis

Features:
  - Eye Contact Analysis (using MediaPipe FaceMesh)
  - Posture Analysis (using MediaPipe Pose)
  - Anxiety Analysis (blink, face touches, jaw tension)
  - Baseline Emotion Analysis (using HSEmotionRecognizer)
  - Updated Emotion Scoring integrating voice pitch features:
       • Extracts pitch standard deviation and pitch range (both normalized)
       • Computes energy variance from RMS energy as a measure of expressiveness
       • Uses a rich set of if-then conditions (penalties and bonuses) to adjust a base score
  - Additional metrics such as Head Nodding Frequency
  - **New Category Scores & Time-Series Data:** • Engagement, Confidence, Stress Level, and Professionalism (tracked each frame)
       • Time-series arrays with timestamps for graphing performance over time

Note: FFmpeg must be installed and available in your PATH.
"""

from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import mediapipe as mp
import librosa

# ================================================================
# App Initialization and Upload Folder Setup
# ================================================================
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================================================================
# CONSTANTS
# ================================================================

# Eye Contact Analysis Constants
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.18
GAZE_OFFSET_THRESHOLD = 0.18
EARLY_FRAME_BUFFER = 10

# Posture Analysis Constant
POSTURE_SHIFT_THRESHOLD = 0.02

# Anxiety Analysis Constants
BLINK_THRESHOLD = 0.2
FACE_TOUCH_DISTANCE = 0.05
JAW_TENSION_THRESHOLD = 0.03

# Baseline Emotion Weights (for reference)
EMOTION_WEIGHTS = {
    'Happiness': 1.0,
    'Surprise': 0.75,
    'Neutral': 0.65,
    'Sadness': 0.65,
    'Fear': 0.3,
    'Anger': 0.2,
    'Disgust': 0.1,
    'Contempt': 0.1
}

# ================================================================
# TEMPLATE FILTERS
# ================================================================
@app.template_filter('base64_encode')
def base64_encode_filter(frame):
    """Encodes a frame (numpy array) as a base64 string for inclusion in HTML."""
    if frame is None:
        return ""
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return ""

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_ear(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) from given eye landmarks.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h_dist = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h_dist)

def analyze_eye_contact(frame, face_mesh, frame_count):
    """
    Analyze eye contact using MediaPipe FaceMesh.
    Returns (good_eye_contact, debug_frame).
    """
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        def get_coords(indices):
            return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)

        left_eye = get_coords(LEFT_EYE)
        right_eye = get_coords(RIGHT_EYE)
        avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        face_center = np.array([np.mean(x_vals) * w, np.mean(y_vals) * h])
        eye_center = np.mean(np.vstack([left_eye, right_eye]), axis=0)
        gaze_offset = np.linalg.norm(eye_center - face_center) / w

        if avg_ear > EAR_THRESHOLD:
            return (gaze_offset < GAZE_OFFSET_THRESHOLD), frame
        return False, frame
    return False, None

def analyze_jaw_tension(face_landmarks):
    """
    Analyze jaw tension based on landmarks.
    Returns True if jaw width is less than threshold.
    """
    jaw_left = np.array([face_landmarks[234].x, face_landmarks[234].y])
    jaw_right = np.array([face_landmarks[454].x, face_landmarks[454].y])
    jaw_width = np.linalg.norm(jaw_left - jaw_right)
    return jaw_width < JAW_TENSION_THRESHOLD

def analyze_posture(frame, pose, last_shoulder_diff, last_hip_diff):
    """
    Analyze posture via MediaPipe Pose landmarks.
    Returns (shift_detected, shoulder_diff, hip_diff, debug_frame).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder  = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_hip      = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        left_hip       = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]

        shoulder_diff = abs(right_shoulder.y - left_shoulder.y)
        hip_diff = abs(right_hip.y - left_hip.y)
        shift_detected = False
        if last_shoulder_diff is not None and last_hip_diff is not None:
            shoulder_change = abs(shoulder_diff - last_shoulder_diff)
            hip_change = abs(hip_diff - last_hip_diff)
            if shoulder_change > POSTURE_SHIFT_THRESHOLD or hip_change > POSTURE_SHIFT_THRESHOLD:
                shift_detected = True
        return shift_detected, shoulder_diff, hip_diff, frame
    return False, None, None, None

def analyze_anxiety(face_mesh, hands, frame):
    """
    Analyzes anxiety signals by detecting:
      - Blinks via EAR,
      - Face touches via hand proximity,
      - Jaw tension via facial landmarks.
    Returns (blink_count, face_touch_count, jaw_tense flag).
    """
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    blink_count = 0
    face_touch_count = 0
    jaw_tense = False

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        eye_coords = np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE])
        ear = calculate_ear(eye_coords)
        if ear < BLINK_THRESHOLD:
            blink_count += 1
        jaw_tense = analyze_jaw_tension(landmarks)
    if hands_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        hand_tip = hands_results.multi_hand_landmarks[0].landmark[8]
        nose_tip = face_results.multi_face_landmarks[0].landmark[1]
        dx = hand_tip.x - nose_tip.x
        dy = hand_tip.y - nose_tip.y
        if np.linalg.norm([dx, dy]) < FACE_TOUCH_DISTANCE:
            face_touch_count += 1
    return blink_count, face_touch_count, jaw_tense

def analyze_pitch(audio_path):
    """
    Analyzes voice pitch from the audio file using librosa.
    Returns (mean_pitch, norm_pitch_std, norm_pitch_range):
      - norm_pitch_std: normalized pitch variability (max ~1.0; typical std ~50 Hz)
      - norm_pitch_range: normalized pitch range (max ~1.0; typical range ~100 Hz)
    """
    try:
        y, sr = librosa.load(audio_path)
        f0_values, voiced_flags, _ = librosa.pyin(y, fmin=80, fmax=400)
        if f0_values is not None:
            valid_f0 = f0_values[~np.isnan(f0_values)]
            if len(valid_f0) > 0:
                mean_pitch = np.mean(valid_f0)
                pitch_std = np.std(valid_f0)
                pitch_range = np.max(valid_f0) - np.min(valid_f0)
                norm_pitch_std = min(pitch_std / 50.0, 1.0)
                norm_pitch_range = min(pitch_range / 100.0, 1.0)
                return mean_pitch, norm_pitch_std, norm_pitch_range
    except Exception as e:
        print(f"Error in analyze_pitch: {e}")
    return None, None, None

def analyze_energy_variance(audio_path):
    """
    Computes expressive energy variance from RMS energy.
    Returns the coefficient of variation (std / mean) of RMS energy (normalized to max 1.0).
    """
    try:
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms) > 0:
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            if mean_rms > 0:
                coeff_var = std_rms / mean_rms
                return round(min(coeff_var, 1.0), 2)
    except Exception as e:
        print(f"Error in analyze_energy_variance: {e}")
    return None

def calculate_emotion_score(audio_emotion_value, norm_pitch_std, norm_pitch_range, facial_emotion_value, energy_variance):
    """
    New emotion scoring function integrating voice pitch features.
    
    All inputs are expected in the range [0.0, 1.0]:
      - audio_emotion_value: higher means more positive/enthusiastic tone.
      - norm_pitch_std: normalized pitch variability (0.0 if monotone to 1.0 if very variable).
      - norm_pitch_range: normalized pitch range (wider is more expressive).
      - facial_emotion_value: proxy from facial landmarks (higher means more positive expressions).
      - energy_variance: coefficient of variation of RMS energy (higher implies more dynamic speech).
    
    Base contributions:
      - 35% audio, 35% facial,
      - 15% from composite voice pitch value (average of norm_pitch_std and norm_pitch_range),
      - 15% from energy variance.
    
    Then a series of if-then adjustments (penalties and bonuses) are applied.
    Finally, the score is clamped to [0.0, 1.0] and returned.
    """
    # Composite voice pitch value (0 to 1)
    voice_pitch_value = (norm_pitch_std + norm_pitch_range) / 2.0

    base_audio = 0.35 * audio_emotion_value
    base_facial = 0.35 * facial_emotion_value
    base_pitch = 0.15 * voice_pitch_value
    base_energy = 0.15 * (energy_variance if energy_variance is not None else 0.5)
    base_score = base_audio + base_facial + base_pitch + base_energy

    penalty = 0.0
    bonus = 0.0

    # Stress Indicators
    if audio_emotion_value < 0.3 and voice_pitch_value > 0.7:
        penalty += 0.15
    if facial_emotion_value > 0.7 and audio_emotion_value < 0.4:
        penalty += 0.10
    if voice_pitch_value < 0.2 and facial_emotion_value > 0.6:
        penalty += 0.05

    # Disengagement
    if audio_emotion_value < 0.4 and facial_emotion_value < 0.4:
        penalty += 0.10
    if voice_pitch_value < 0.3 and 0.3 < audio_emotion_value < 0.6:
        penalty += 0.05

    # Over-enthusiasm / Unnatural Excitement
    if voice_pitch_value > 0.8 and audio_emotion_value > 0.8 and facial_emotion_value < 0.5:
        penalty += 0.10
    if voice_pitch_value > 0.9 and 0.4 < facial_emotion_value < 0.6:
        penalty += 0.05

    # Consistency Bonuses
    if audio_emotion_value > 0.7 and facial_emotion_value > 0.7 and 0.4 < voice_pitch_value < 0.8:
        bonus += 0.15
    if all(0.45 < v < 0.65 for v in (audio_emotion_value, facial_emotion_value, voice_pitch_value)):
        bonus += 0.10
    if 0.45 < voice_pitch_value < 0.55 and audio_emotion_value > 0.8 and facial_emotion_value > 0.8:
        bonus += 0.10

    # Mixed Signals
    if audio_emotion_value > 0.75 and facial_emotion_value < 0.5 and 0.3 < voice_pitch_value < 0.7:
        penalty += 0.08
    if facial_emotion_value > 0.75 and 0.4 < audio_emotion_value < 0.6 and voice_pitch_value < 0.4:
        penalty += 0.07
    if facial_emotion_value < 0.4 and 0.4 < audio_emotion_value < 0.7 and voice_pitch_value > 0.7:
        penalty += 0.07
    if 0.4 < facial_emotion_value < 0.6 and audio_emotion_value < 0.4 and 0.4 < voice_pitch_value < 0.6:
        penalty += 0.05

    # Additional bonus for dynamic energy (if energy variance is high)
    if energy_variance is not None and energy_variance > 0.5:
        bonus += 0.10

    adjusted = base_score - penalty + bonus
    final = max(0.0, min(1.0, adjusted))
    return round(final, 3)

def analyze_video(video_path, norm_pitch_std, norm_pitch_range, energy_variance, audio_filepath):
    """
    Processes the video file frame-by-frame to compute various metrics:
      - Baseline emotion analysis (via HSEmotionRecognizer)
      - Eye contact, posture, anxiety, head nodding, etc.
    
    Additionally, gathers time-series data (per frame) for four new parameters:
      Engagement, Confidence, Stress Level, and Professionalism.
    
    Returns a dictionary with overall scores, time-series arrays, and other metrics.
    """
    # Initialize MediaPipe models
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    emotion_counts = {}
    frame_count = 0
    analyzed_frames = 0
    eye_avoidance_count = 0
    posture_shift_count = 0
    blink_count = 0
    face_touch_count = 0
    jaw_tension_count = 0
    head_nod_count = 0
    previous_nose_y = None
    last_direction = None

    # Time-series arrays for new parameters
    times = []
    engagement_ts = []
    confidence_ts = []
    stress_ts = []
    professionalism_ts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = frame_count / fps
        times.append(round(current_time, 2))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Baseline Emotion Analysis using HSEmotionRecognizer
        face_results_detect = face_detector.process(rgb)
        if face_results_detect.detections:
            ih, iw, _ = frame.shape
            for det in face_results_detect.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * iw)
                y1 = int(bbox.ymin * ih)
                w_box = int(bbox.width * iw)
                h_box = int(bbox.height * ih)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x1 + w_box), min(ih, y1 + h_box)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                face_roi = cv2.resize(face_roi, (224, 224))
                emotion, _ = fer.predict_emotions(face_roi, logits=False)
                emotion_label = emotion if isinstance(emotion, str) else emotion[0]
                emotion_counts[emotion_label] = emotion_counts.get(emotion_label, 0) + 1
                analyzed_frames += 1
                break
        
        # Eye Contact via FaceMesh (after early buffer)
        if frame_count > EARLY_FRAME_BUFFER:
            good_contact, _ = analyze_eye_contact(frame, face_mesh, frame_count)
            if not good_contact:
                eye_avoidance_count += 1
        
        # Posture via Pose
        shift_detected, shoulder_diff, hip_diff, _ = analyze_posture(frame, pose, None, None)
        if shift_detected:
            posture_shift_count += 1
        
        # Anxiety via FaceMesh and Hands
        b_count, ft_count, jaw_tense = analyze_anxiety(face_mesh, hands, frame)
        blink_count += b_count
        face_touch_count += ft_count
        if jaw_tense:
            jaw_tension_count += 1
        
        # Head Nodding Frequency (using nose tip from FaceMesh)
        face_mesh_results = face_mesh.process(rgb)
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0].landmark
            current_nose_y = landmarks[1].y
            if previous_nose_y is not None:
                delta = current_nose_y - previous_nose_y
                if abs(delta) > 0.02:
                    direction = "down" if delta > 0 else "up"
                    if last_direction is not None and direction != last_direction:
                        head_nod_count += 1
                    last_direction = direction
            previous_nose_y = current_nose_y

        # --- Compute Per-Frame Derived Metrics for Time-Series ---
        # Compute Eye Contact Score (0-10) using face mesh data (proxy based on gaze offset)
        if frame_count > EARLY_FRAME_BUFFER and face_mesh_results.multi_face_landmarks:
            landmarks_temp = face_mesh_results.multi_face_landmarks[0].landmark
            h_frame, w_frame = frame.shape[:2]
            x_vals = [lm.x for lm in landmarks_temp]
            y_vals = [lm.y for lm in landmarks_temp]
            face_center = np.array([np.mean(x_vals) * w_frame, np.mean(y_vals) * h_frame])
            def get_coords(indices):
                return np.array([[landmarks_temp[i].x * w_frame, landmarks_temp[i].y * h_frame] for i in indices], dtype=np.float32)
            left_eye = get_coords(LEFT_EYE)
            right_eye = get_coords(RIGHT_EYE)
            eye_center = np.mean(np.vstack([left_eye, right_eye]), axis=0)
            gaze_offset = np.linalg.norm(eye_center - face_center) / w_frame
            eye_contact_frame = max(0, min(10, 10 * (1 - (gaze_offset / GAZE_OFFSET_THRESHOLD))))
        else:
            eye_contact_frame = 0

        # Posture Score per frame (0-10): for simplicity, if pose detected, assign 10 else 0.
        posture_frame = 10 if pose.process(rgb).pose_landmarks else 0

        # Anxiety proxy per frame using blink detection via EAR from face mesh
        anxiety_results = face_mesh.process(rgb)
        if anxiety_results.multi_face_landmarks:
            landmarks_anxiety = anxiety_results.multi_face_landmarks[0].landmark
            eye_coords_anxiety = np.array([[landmarks_anxiety[i].x, landmarks_anxiety[i].y] for i in LEFT_EYE])
            ear_frame = calculate_ear(eye_coords_anxiety)
            anxiety_frame = 5 if ear_frame < BLINK_THRESHOLD else 10
        else:
            anxiety_frame = 0

        # Facial Emotion proxy per frame: using average landmark y value (scaled to 0-10)
        if face_mesh_results.multi_face_landmarks:
            avg_y = np.mean([lm.y for lm in face_mesh_results.multi_face_landmarks[0].landmark])
            facial_emotion_frame = max(0, min(10, (1 - avg_y) * 10))
        else:
            facial_emotion_frame = 5

        # Derived New Metrics per frame:
        # Engagement: 60% eye contact + 40% posture (scale: 0-10)
        engagement_frame = round(0.6 * eye_contact_frame + 0.4 * posture_frame, 2)
        # Confidence: 50% eye contact + 40% (norm_pitch_range*10) + 10% posture
        confidence_frame = round(0.5 * eye_contact_frame + 0.4 * (norm_pitch_range * 10) + 0.1 * posture_frame, 2)
        # Stress: Inverse relation — higher anxiety & lower facial emotion mean more stress
        stress_frame = round(10 - (((10 - anxiety_frame) * 0.6) + (facial_emotion_frame * 0.4)), 2)
        # Professionalism: 50% posture + 50% (energy_variance*10)
        professionalism_frame = round(0.5 * posture_frame + 0.5 * (energy_variance * 10), 2)

        engagement_ts.append(engagement_frame)
        confidence_ts.append(confidence_frame)
        stress_ts.append(stress_frame)
        professionalism_ts.append(professionalism_frame)
    
    cap.release()
    
    # --- Global Score Calculations ---
    if frame_count > EARLY_FRAME_BUFFER:
        effective_frames = frame_count - EARLY_FRAME_BUFFER
        eye_contact_ratio = eye_avoidance_count / effective_frames
        eye_contact_score = round((1 - eye_contact_ratio) * 10, 2)
    else:
        eye_contact_score = 0

    if frame_count > 0:
        shift_ratio = posture_shift_count / frame_count
        posture_score = round(max(0, 10 - (shift_ratio * 15)), 2)
    else:
        posture_score = 0

    if analyzed_frames > 0:
        weighted_sum = sum(emotion_counts.get(e, 0) * EMOTION_WEIGHTS.get(e, 0) for e in emotion_counts)
        base_emotion_score = (weighted_sum / analyzed_frames) * 10
        dominant_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        dominant_share = sum([count for _, count in dominant_emotions]) / analyzed_frames
        consistency_boost = 1 + min(0.3, max(0, dominant_share - 0.6))
        positive_emotions = ['Happiness', 'Surprise']
        positive_ratio = sum(emotion_counts.get(e, 0) for e in positive_emotions) / analyzed_frames
        if positive_ratio > 0.6:
            consistency_boost += 0.1
        old_emotion_score = round(min(10, base_emotion_score * consistency_boost), 2)
    else:
        old_emotion_score = 5.0

    # Updated Emotion Score using voice pitch features computed from a representative frame
    cap = cv2.VideoCapture(video_path)
    updated_emotion_score = 5.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            facial_emotion_value = np.mean([lm.y for lm in face_results.multi_face_landmarks[0].landmark])
            audio_emotion_value = 0.6  # Placeholder from audio sentiment analysis
            updated_emotion_score = calculate_emotion_score(audio_emotion_value,
                                                            norm_pitch_std,
                                                            norm_pitch_range,
                                                            facial_emotion_value,
                                                            energy_variance)
    cap.release()
    final_emotion_score = updated_emotion_score

    if frame_count > 0:
        anxiety_ratio = (blink_count + face_touch_count + eye_avoidance_count + jaw_tension_count) / frame_count
        anxiety_score = round((1 - anxiety_ratio) * 10, 2)
    else:
        anxiety_score = 0

    # Final Overall Score: Weights - Emotion 40%, Eye Contact 20%, Posture 20%, Anxiety 20%
    final_score = round((final_emotion_score * 0.4 +
                         eye_contact_score * 0.2 +
                         posture_score * 0.2 +
                         anxiety_score * 0.2), 2)

    # Compute average values for new categories from time-series data
    engagement_final = round(np.mean(engagement_ts) if engagement_ts else 0, 2)
    confidence_final = round(np.mean(confidence_ts) if confidence_ts else 0, 2)
    stress_final = round(np.mean(stress_ts) if stress_ts else 0, 2)
    professionalism_final = round(np.mean(professionalism_ts) if professionalism_ts else 0, 2)

    print(f"DEBUG: times: {times[:5]}...")  # Print first 5 elements
    print(f"DEBUG: engagement_ts: {engagement_ts[:5]}...")
    print(f"DEBUG: confidence_ts: {confidence_ts[:5]}...")
    print(f"DEBUG: stress_ts: {stress_ts[:5]}...")
    print(f"DEBUG: professionalism_ts: {professionalism_ts[:5]}...")


    return {
        'frame_count': frame_count,
        'analyzed_frames': analyzed_frames,
        'emotions': emotion_counts,
        'eye_avoidance_count': eye_avoidance_count,
        'eye_contact_score': eye_contact_score,
        'posture_shift_count': posture_shift_count,
        'posture_score': posture_score,
        'old_emotion_score': old_emotion_score,
        'emotion_score': final_emotion_score,
        'blink_count': blink_count,
        'face_touch_count': face_touch_count,
        'jaw_tension_count': jaw_tension_count,
        'anxiety_score': anxiety_score,
        'final_score': final_score,
        'norm_pitch_std': norm_pitch_std,
        'norm_pitch_range': norm_pitch_range,
        'energy_variance': energy_variance,
        'head_nod_count': head_nod_count,
        # New Category Averages
        'engagement_final': engagement_final,
        'confidence_final': confidence_final,
        'stress_final': stress_final,
        'professionalism_final': professionalism_final,
        # Time-series arrays for graphing
        'times': times,
        'engagement_ts': engagement_ts,
        'confidence_ts': confidence_ts,
        'stress_ts': stress_ts,
        'professionalism_ts': professionalism_ts
    }

# ================================================================
# FLASK ROUTES
# ================================================================
@app.route('/', methods=['GET'])
def index():
    """Render the home page (webcam interface)."""
    return render_template('index.html')

@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    """
    Receives the recorded webcam video, saves it,
    converts it to audio using ffmpeg, extracts pitch and energy metrics,
    then performs full video analysis using the updated emotion scoring system
    along with new category scores and time-series data for graphing.
    Finally, renders the results page with all computed metrics.
    """
    file = request.files['video']
    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Convert video to audio using ffmpeg (ensure ffmpeg is installed)
        audio_filepath = filepath.replace('.webm', '.wav')
        ffmpeg_command = f'ffmpeg -i "{filepath}" -ar 16000 "{audio_filepath}" -y'
        os.system(ffmpeg_command)
        
        # Analyze audio pitch: get mean_pitch, norm_pitch_std, norm_pitch_range
        mean_pitch, norm_pitch_std, norm_pitch_range = analyze_pitch(audio_filepath)
        if mean_pitch is None or norm_pitch_std is None or norm_pitch_range is None:
            mean_pitch, norm_pitch_std, norm_pitch_range = 200, 0.5, 0.5
        
        # Analyze energy variance from audio (RMS)
        energy_variance = analyze_energy_variance(audio_filepath)
        if energy_variance is None:
            energy_variance = 0.5
        
        # Run full video analysis with the extracted metrics
        results = analyze_video(filepath, norm_pitch_std, norm_pitch_range, energy_variance, audio_filepath)
        return render_template('result.html', **results)
    return "No video received", 400

# ================================================================
# RUN THE APP
# ================================================================
if __name__ == '__main__':
    app.run(debug=True)
