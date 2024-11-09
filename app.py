import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import math
import warnings

# Suppress TensorFlow and protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warnings from TensorFlow
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype.*")

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models and scalers for models 1-5
models = {
    "1": tf.keras.models.load_model('model1.h5'),
    "2": tf.keras.models.load_model('model2.h5'),
    "3": tf.keras.models.load_model('model3.h5'),
    "4": tf.keras.models.load_model('model4.h5'),
    "5": tf.keras.models.load_model('model5.h5')
}

scalers = {
    "1": joblib.load('scaler1.joblib'),
    "2": joblib.load('scaler2.joblib'),
    "3": joblib.load('scaler3.joblib'),
    "4": joblib.load('scaler4.joblib'),
    "5": joblib.load('scaler5.joblib')
}

# Define the model-specific thresholds
thresholds = {
    "1": 0.5,
    "2": 0.6,
    "3": 0.55,
    "4": 0.5,
    "5": 0.6
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the body parts indices in MediaPipe Pose
BODY_PARTS = {
    'left_shoulder': 11, 'right_shoulder': 12, 
    'left_elbow': 13, 'right_elbow': 14, 
    'left_wrist': 15, 'right_wrist': 16, 
    'left_hip': 23, 'right_hip': 24, 
    'left_knee': 25, 'right_knee': 26, 
    'left_ankle': 27, 'right_ankle': 28
}

# Helper Functions for Model 0 (Pose Detection, Angles, Heights)
def detect_pose(image):
    results = pose.process(image)

    if results.pose_landmarks:
        left_leg_angle = calculate_angle(image, results.pose_landmarks, 23, 25, 27)
        right_leg_angle = calculate_angle(image, results.pose_landmarks, 24, 26, 28)

        left_shoulder_to_elbow_height = calculate_height(image, results.pose_landmarks, 11, 13)
        left_shoulder_to_wrist_height = calculate_height(image, results.pose_landmarks, 11, 15)
        right_shoulder_to_elbow_height = calculate_height(image, results.pose_landmarks, 12, 14)
        right_shoulder_to_wrist_height = calculate_height(image, results.pose_landmarks, 12, 16)

        return {
            "left_leg_angle": left_leg_angle,
            "right_leg_angle": right_leg_angle,
            "left_shoulder_to_elbow_height": left_shoulder_to_elbow_height,
            "left_shoulder_to_wrist_height": left_shoulder_to_wrist_height,
            "right_shoulder_to_elbow_height": right_shoulder_to_elbow_height,
            "right_shoulder_to_wrist_height": right_shoulder_to_wrist_height
        }
    return None

def calculate_angle(image, landmarks, point1, point2, point3):
    point1_x = landmarks.landmark[point1].x * image.shape[1]
    point1_y = landmarks.landmark[point1].y * image.shape[0]
    point2_x = landmarks.landmark[point2].x * image.shape[1]
    point2_y = landmarks.landmark[point2].y * image.shape[0]
    point3_x = landmarks.landmark[point3].x * image.shape[1]
    point3_y = landmarks.landmark[point3].y * image.shape[0]

    v1 = [point1_x - point2_x, point1_y - point2_y]
    v2 = [point3_x - point2_x, point3_y - point2_y]

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle = math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))

    return angle

def calculate_height(image, landmarks, point1, point2):
    point1_y = landmarks.landmark[point1].y * image.shape[0]
    point2_y = landmarks.landmark[point2].y * image.shape[0]
    height = abs(point2_y - point1_y)
    return height

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return detect_pose(image_rgb)

# Analyze function for model 0 (Pose Analysis)
def analyze_pose(input_path):
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # Handle image input
        image_results = process_image(input_path)
        if image_results:
            pose_correct = True
            incorrect_reasons = []

            # Check left leg angle
            if not (100 <= image_results['left_leg_angle'] <= 125):
                pose_correct = False
                incorrect_reasons.append(f"Left leg angle is out of range")

            # Check right leg angle
            if not (100 <= image_results['right_leg_angle'] <= 125):
                pose_correct = False
                incorrect_reasons.append(f"Right leg angle is out of range")

            # Check shoulder to elbow height
            if not (-50 <= image_results['left_shoulder_to_elbow_height'] <= 50):
                pose_correct = False
                incorrect_reasons.append(f"Left shoulder to elbow height is out of range")
            if not (-50 <= image_results['right_shoulder_to_elbow_height'] <= 50):
                pose_correct = False
                incorrect_reasons.append(f"Right shoulder to elbow height is out of range")

            # Check shoulder to wrist height
            if not (-50 <= image_results['left_shoulder_to_wrist_height'] <= 50):
                pose_correct = False
                incorrect_reasons.append(f"Left shoulder to wrist height is out of range")
            if not (-50 <= image_results['right_shoulder_to_wrist_height'] <= 50):
                pose_correct = False
                incorrect_reasons.append(f"Right shoulder to wrist height is out of range")

            # Return detailed result
            if pose_correct:
                return {"status": "correct", "incorrect_parts": incorrect_reasons}
            else:
                return {"status": "incorrect", "incorrect_parts": incorrect_reasons}
    
    # For videos or invalid image formats:
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Handle video input
        return analyze_video_pose(input_path)
    
    return {"status": "error", "message": "Could not process image or video"}

# Analyze video function for models 1-5 (TensorFlow predictions)
def extract_keypoints_from_frame(frame):
    results = pose.process(frame)
    keypoints = []
    if results.pose_landmarks:
        h, w, _ = frame.shape
        for part_name, idx in BODY_PARTS.items():
            keypoints.append(results.pose_landmarks.landmark[idx].x * w)
            keypoints.append(results.pose_landmarks.landmark[idx].y * h)
        return keypoints
    return None

def process_video(video_path, scaler):
    cap = cv2.VideoCapture(video_path)
    keypoints_all_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints_from_frame(frame)
        if keypoints:
            keypoints_all_frames.append(keypoints)

    cap.release()

    if keypoints_all_frames:
        keypoints_all_frames = scaler.transform(keypoints_all_frames)
        keypoints_all_frames = keypoints_all_frames.reshape((keypoints_all_frames.shape[0], 1, keypoints_all_frames.shape[1]))
        return keypoints_all_frames
    return None

def analyze_video(video_path, model, scaler, threshold):
    keypoints_all_frames = process_video(video_path, scaler)
    if keypoints_all_frames is not None:
        predictions = model.predict(keypoints_all_frames)
        incorrect_parts = {part: 0 for part in BODY_PARTS.keys()}

        for i, pred in enumerate(predictions):
            if pred < threshold:
                keypoints = keypoints_all_frames[i].reshape(-1, 2)
                for j, (x, y) in enumerate(keypoints):
                    if x < 0.1 or y < 0.1:
                        part_name = list(BODY_PARTS.keys())[j // 2]
                        incorrect_parts[part_name] += 1

        average_prediction = np.mean(predictions)
        if average_prediction > threshold:
            return {"status": "correct", "incorrect_parts": incorrect_parts}
        else:
            return {"status": "incorrect", "incorrect_parts": incorrect_parts}
    else:
        return {"status": "error", "message": "Could not process video"}
    
def analyze_video_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    correct_frames = 0
    incorrect_reasons = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame of the video to extract pose landmarks
        image_results = detect_pose(frame)
        if image_results:
            total_frames += 1
            pose_correct = True
            frame_incorrect_reasons = []

            # Check for pose correctness for each frame
            # Check left leg angle
            if not (100 <= image_results['left_leg_angle'] <= 125):
                pose_correct = False
                frame_incorrect_reasons.append(f"Left leg angle is out of range")

            # Check right leg angle
            if not (100 <= image_results['right_leg_angle'] <= 125):
                pose_correct = False
                frame_incorrect_reasons.append(f"Right leg angle is out of range")

            # Check shoulder to elbow height
            if not (-50 <= image_results['left_shoulder_to_elbow_height'] <= 50):
                pose_correct = False
                frame_incorrect_reasons.append(f"Left shoulder to elbow height is out of range")
            if not (-50 <= image_results['right_shoulder_to_elbow_height'] <= 50):
                pose_correct = False
                frame_incorrect_reasons.append(f"Right shoulder to elbow height is out of range")

            # Check shoulder to wrist height
            if not (-50 <= image_results['left_shoulder_to_wrist_height'] <= 50):
                pose_correct = False
                frame_incorrect_reasons.append(f"Left shoulder to wrist height is out of range")
            if not (-50 <= image_results['right_shoulder_to_wrist_height'] <= 50):
                pose_correct = False
                frame_incorrect_reasons.append(f"Right shoulder to wrist height is out of range")

            # If the frame is correct, increment the correct frames count
            if pose_correct:
                correct_frames += 1

            # Append the reasons for the current frame if it's incorrect
            if frame_incorrect_reasons:
                incorrect_reasons.extend(frame_incorrect_reasons)

    cap.release()

    # After processing all frames, calculate the average result
    if total_frames > 0:
        average_correct = (correct_frames / total_frames) * 100
        # You can choose the threshold for the result, for example, 80% correct frames to be considered "correct"
        if average_correct >= 80:
            return {"status": "correct", "incorrect_parts": list(dict.fromkeys(incorrect_reasons))}
        else:
            return {"status": "incorrect", "incorrect_parts": list(dict.fromkeys(incorrect_reasons))}
    else:
        return {"status": "error", "message": "No frames to process"}


# API Route to accept video uploads and process based on model number
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files and 'image' not in request.files or 'modelnumber' not in request.form:
        return jsonify({"status": "error", "message": "Video/image file and model number are required"}), 400
    
    file = request.files.get('video') or request.files.get('image')
    modelnumber = request.form['modelnumber']

    # Validate model number
    if modelnumber not in models and modelnumber != "0":
        return jsonify({"status": "error", "message": "Invalid model number"}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Model 0 (Pose Analysis)
    if modelnumber == "0":
        result = analyze_pose(file_path)
    
    # Models 1-5 (TensorFlow-based analysis)
    else:
        if modelnumber not in models or modelnumber not in scalers:
            return jsonify({"status": "error", "message": "Invalid model number"}), 400
        
        model = models[modelnumber]
        scaler = scalers[modelnumber]
        threshold = thresholds[modelnumber]
        result = analyze_video(file_path, model, scaler, threshold)

    # Remove the file after processing
    os.remove(file_path)

    return jsonify(result)

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the Flask app
    app.run(debug=True)
