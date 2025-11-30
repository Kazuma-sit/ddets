# utils.py
import cv2
import dlib
import numpy as np
from config import SHAPE_PREDICTOR_PATH

# --- Global Initializations ---
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def get_face_bbox(frame):
    """Function to get face bounding box using OpenFace."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces) > 0:
        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        x_min = min([p.x for p in landmarks.parts()])
        x_max = max([p.x for p in landmarks.parts()])
        y_min = min([p.y for p in landmarks.parts()])
        y_max = max([p.y for p in landmarks.parts()])
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    return None

def interpolate_nan(scores):
    """Function to interpolate NaN values."""
    scores = np.array(scores, dtype=float) # Ensure scores are float to handle NaN
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            if i > 0:
                scores[i] = scores[i - 1]
            else:
                scores[i] = 0 # Default to 0 if the first value is NaN
    return scores

def logits_to_probability(logits):
    """Function to convert logits to probabilities."""
    return 1 / (1 + np.exp(-logits))