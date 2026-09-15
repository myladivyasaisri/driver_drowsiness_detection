import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np
import base64

st.title("Driver Drowsiness Detection System")
st.subheader("Edge-AI Safety System for Real-Time Fatigue Monitoring")

# 1. HTML5 Audio Setup for Browser Speakers
# Online free alarm sound link (beep sound)
ALARM_URL = "https://soundjay.com"

def play_alarm_in_browser():
    """Plays alarm sound directly through the user's browser speakers"""
    audio_html = f"""
        <audio autoplay loop>
            <source src="{ALARM_URL}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# 2. EAR & MAR Calculations
def calculate_ear(eye):
    # Vector arithmetic simulation
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))   # p51 - p59
    B = np.linalg.norm(np.array(mouth[5]) - np.array(mouth[7]))   # p53 - p57
    C = np.linalg.norm(np.array(mouth[1]) - np.array(mouth[11]))  # p49 - p55
    return (A + B) / (2.0 * C)

# 3. Initialize Dlib
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("⚠️ models/ folder lo shape_predictor_68_face_landmarks.dat file ledu!")

# Using Streamlit session state to pass state variables from thread to UI layer
if "drowsy_state" not in st.session_state:
    st.session_state.drowsy_state = False

class DrowsinessTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        is_drowsy_now = False
        
        for face in faces:
            landmarks = predictor(gray, face)
            points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            
            left_eye = points[36:42]
            right_eye = points[42:48]
            mouth = points[48:68]
            
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)
            
            if ear < 0.25 or mar > 0.60:
                is_drowsy_now = True
                cv2.putText(img, "DROWSY (CRITICAL!)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, "STATUS: AWAKE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        st.session_state.drowsy_state = is_drowsy_now
        return img

# Web-cam view container
webrtc_streamer(key="drowsiness-detection", video_transformer_factory=DrowsinessTransformer)

# 4. Sound Engine Trigger
# Trigger alarm injection onto Web interface when condition state flag trips
if st.session_state.drowsy_state:
    play_alarm_in_browser()
