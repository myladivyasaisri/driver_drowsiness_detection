import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np

st.title("Driver Drowsiness Detection System")
st.subheader("Edge-AI Safety System for Real-Time Fatigue Monitoring")

# 1. EAR & MAR Calculations
def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[13]) - np.array(mouth[19])) # p51 - p59
    B = np.linalg.norm(np.array(mouth[15]) - np.array(mouth[17])) # p53 - p57
    C = np.linalg.norm(np.array(mouth[11]) - np.array(mouth[12])) # p49 - p55
    return (A + B) / (2.0 * C)

# 2. Initialize Dlib
detector = dlib.get_frontal_face_detector()

# Missing model file check
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except Exception as e:
    st.error("⚠️ models/ folder లో shape_predictor_68_face_landmarks.dat ఫైల్ లేదు! దయచేసి దాన్ని గిట్‌హబ్‌లో అప్‌లోడ్ చేయండి.")

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))
                # Unified Chrome-Green Mesh (BGR: 0, 255, 0)
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            
            # Extract Eye and Mouth Coordinates
            left_eye = points[36:42]
            right_eye = points[42:48]
            mouth = points[48:68]
            
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)
            
            # Status Alerts logic & Dynamic HUD Display
            if ear < 0.25:
                cv2.putText(img, "DROWSY (CRITICAL!)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif mar > 0.60:
                cv2.putText(img, "YAWNING DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                cv2.putText(img, "STATUS: AWAKE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return img

webrtc_streamer(key="drowsiness-detection", video_transformer_factory=DrowsinessTransformer)
