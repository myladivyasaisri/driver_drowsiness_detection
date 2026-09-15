import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np

st.title("Driver Drowsiness Detection System")
st.subheader("An Edge-AI Safety System for Real-Time Fatigue Monitoring")

# Audio URLs for browser playback
DROWSY_ALARM = "https://soundjay.com"  # Aggressive beep
YAWN_ALARM = "https://soundjay.com"    # Different tone beep

def play_sound(url):
    """Injects HTML5 audio player directly into user's browser"""
    audio_html = f"""
        <audio autoplay loop>
            <source src="{url}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# EAR & MAR Math Vector Equations
def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[13]) - np.array(mouth[19])) # p51 - p59
    B = np.linalg.norm(np.array(mouth[15]) - np.array(mouth[17])) # p53 - p57
    C = np.linalg.norm(np.array(mouth[12]) - np.array(mouth[16])) # p49 - p55
    return (A + B) / (2.0 * C)

# Initialize Dlib Face Mesh Framework
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("⚠️ models/ folder lo shape_predictor_68_face_landmarks.dat file ledu!")

# Manage global interface states across Streamlit frames
if "current_status" not in st.session_state:
    st.session_state.current_status = "AWAKE"

class DrowsinessTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        status = "AWAKE"
        
        for face in faces:
            landmarks = predictor(gray, face)
            points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append((x, y))
                # Draw Unified Chrome-Green Mesh
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            
            left_eye = points[36:42]
            right_eye = points[42:48]
            mouth = points[48:68]
            
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)
            
            # Exact status rules configuration logic
            if ear < 0.25:
                status = "DROWSY"
                cv2.putText(img, "STATUS: DROWSY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif mar > 0.60:
                status = "YAWNING"
                cv2.putText(img, "STATUS: YAWNING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                status = "AWAKE"
                cv2.putText(img, "STATUS: AWAKE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        st.session_state.current_status = status
        return img

# Web stream runtime container
webrtc_streamer(key="drowsiness-detection", video_transformer_factory=DrowsinessTransformer)

# Trigger targeted browser audio based on precise runtime flags
if st.session_state.current_status == "DROWSY":
    play_sound(DROWSY_ALARM)
elif st.session_state.current_status == "YAWNING":
    play_sound(YAWN_ALARM)
