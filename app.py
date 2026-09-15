import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import dlib
import numpy as np

st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Interface")

# Audio alert elements using HTML5
DROWSY_ALARM = "https://soundjay.com"
YAWN_ALARM = "https://soundjay.com"

def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))
    B = np.linalg.norm(np.array(mouth[5]) - np.array(mouth[7]))
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    return (A + B) / (2.0 * C)

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("Missing model file inside models/ folder!")

class DrowsinessTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        status = "AWAKE"
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
                status = "DROWSY"
                color = (0, 0, 255)
            elif mar > 0.60:
                status = "YAWNING"
                color = (0, 165, 255)
            else:
                status = "AWAKE"
                color = (0, 255, 0)
            
            cv2.putText(img, f"STATUS: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(img, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return img

webrtc_streamer(
    key="drowsiness-detection", 
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True
)
