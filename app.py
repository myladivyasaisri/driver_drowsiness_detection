import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import dlib
import numpy as np

st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Interface")

# Direct online audio URL links for browser playback
DROWSY_ALARM_URL = "https://soundjay.com"
YAWN_ALARM_URL = "https://soundjay.com"

def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[11]))
    B = np.linalg.norm(np.array(mouth[5]) - np.array(mouth[9]))
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    return (A + B) / (2.0 * C)

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("Missing model file inside models/ folder!")

# Audio placeholders to inject sound dynamically into the browser
drowsy_audio_placeholder = st.empty()
yawn_audio_placeholder = st.empty()

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.status = "AWAKE"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        local_status = "AWAKE"
        color = (0, 255, 0)
        
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
            
            if ear < 0.25:
                local_status = "DROWSY"
                color = (0, 0, 255)
            elif mar > 0.60:
                local_status = "YAWNING"
                color = (0, 165, 255)
            else:
                local_status = "AWAKE"
                color = (0, 255, 0)
            
            cv2.putText(img, f"STATUS: {local_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        self.status = local_status
        return img

webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection", 
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Active status listener loop to inject audio into the main web UI layer
if webrtc_ctx.video_transformer:
    current_condition = webrtc_ctx.video_transformer.status
    
    if current_condition == "DROWSY":
        drowsy_audio_placeholder.markdown(
            f'<audio autoplay loop><source src="{DROWSY_ALARM_URL}" type="audio/mp3"></audio>', 
            unsafe_allow_html=True
        )
        yawn_audio_placeholder.empty()
    elif current_condition == "YAWNING":
        yawn_audio_placeholder.markdown(
            f'<audio autoplay loop><source src="{YAWN_ALARM_URL}" type="audio/mp3"></audio>', 
            unsafe_allow_html=True
        )
        drowsy_audio_placeholder.empty()
    else:
        drowsy_audio_placeholder.empty()
        yawn_audio_placeholder.empty()
