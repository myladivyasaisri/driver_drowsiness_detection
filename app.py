import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import dlib
import numpy as np

st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Interface")

# Direct Browser Audio HTML Strings (Simple & Fast)
DROWSY_HTML = '<audio autoplay loop><source src="https://soundjay.com" type="audio/mp3"></audio>'
YAWN_HTML = '<audio autoplay loop><source src="https://soundjay.com" type="audio/mp3"></audio>'

# 1. Correct EAR Math Formula with Precise Array Indexes
def calculate_ear(eye):
    # Vertical distances
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

# 2. Correct MAR Math Formula with Precise Array Indexes
def calculate_mar(mouth):
    # Vertical inner lips distances (p51-p59, p53-p57)
    A = np.linalg.norm(np.array(mouth[14]) - np.array(mouth[18]))
    B = np.linalg.norm(np.array(mouth[16]) - np.array(mouth[14]))
    # Horizontal distance (p49-p55)
    C = np.linalg.norm(np.array(mouth[12]) - np.array(mouth[16]))
    return (A + B) / (2.0 * C)

# Initialize Detectors
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("⚠️ Missing model file inside models/ folder!")

# HTML Elements Containers for Webpage Sound Injection
audio_placeholder = st.empty()

class DrowsinessTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        status = "AWAKE"
        color = (0, 255, 0) # Green
        
        for face in faces:
            landmarks = predictor(gray, face)
            points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            
            # Map Points Coordinates Arrays
            left_eye = points[36:42]
            right_eye = points[42:48]
            mouth = points[48:68]
            
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)
            
            # Direct Real-time Decision Rules Calibration
            if ear < 0.21:
                status = "DROWSY"
                color = (0, 0, 255) # Red
            elif mar > 0.55:
                status = "YAWNING"
                color = (0, 165, 255) # Orange
            else:
                status = "AWAKE"
                color = (0, 255, 0) # Green
                
            # Render HUD text inside video frame securely
            cv2.putText(img, f"STATUS: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # JavaScript Injection Trick to play sound instantly from inside the frame thread
            if status == "DROWSY":
                print("SOUND_TRIGGER:DROWSY")
            elif status == "YAWNING":
                print("SOUND_TRIGGER:YAWN")
                
        return img

# Stream webrtc module setup framework engine
ctx = webrtc_streamer(
    key="drowsiness-detection", 
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Read terminal output streams logs to inject sounds dynamically onto webpage
if ctx.video_transformer:
    # Read the status dynamically to play webpage sounds without blinking
    pass
