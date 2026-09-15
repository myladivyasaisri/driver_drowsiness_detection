import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np

st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Link")

# EAR & MAR Math Formulas (Direct ga ఇక్కడే రాసేసాము)
def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10])) # p51 - p59
    B = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))  # p53 - p57
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))  # p49 - p55
    return (A + B) / (2.0 * C)

# Dlib detectors setup
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
        
        for face in faces:
            # Mee custom tracking variables & HUD parameters ikada run avthayi
            pass
            
        return img

# Web-cam connection framework
webrtc_streamer(key="drowsiness-detection", video_transformer_factory=DrowsinessTransformer)
