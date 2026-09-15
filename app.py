import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import dlib
import numpy as np

st.set_page_config(page_title="ADAS Infrastructure Dashboard", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Interface")

# Audio alert links for browser output
DROWSY_ALARM_URL = "https://soundjay.com"
YAWN_ALARM_URL = "https://soundjay.com"

# Initialize global configuration states
if "alarm_state" not in st.session_state:
    st.session_state.alarm_state = "none"

def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye) - np.array(eye))
    B = np.linalg.norm(np.array(eye) - np.array(eye))
    C = np.linalg.norm(np.array(eye) - np.array(eye))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth) - np.array(mouth))
    B = np.linalg.norm(np.array(mouth) - np.array(mouth))
    C = np.linalg.norm(np.array(mouth) - np.array(mouth))
    return (A + B) / (2.0 * C)

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("Missing model file inside models/ folder!")

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.eye_counter = 0
        self.yawn_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        driver_status = "AWAKE"
        status_color = (0, 255, 0) 
        local_alarm = "none"

        if len(rects) == 0:
            cv2.putText(img, "STATUS: SCANNING ROAD", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        for rect in rects:
            shape = predictor(gray, rect)
            points = []
            for n in range(0, 68):
                points.append((shape.part(n).x, shape.part(n).y))
            
            leftEye = points[36:42]
            rightEye = points[42:48]
            mouth = points[48:68]

            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
            mar = calculate_mar(mouth)

            cv2.polylines(img, [np.array(leftEye)], True, (0, 255, 0), 1)
            cv2.polylines(img, [np.array(rightEye)], True, (0, 255, 0), 1)
            cv2.polylines(img, [np.array(mouth)], True, (0, 255, 0), 1)

            # Your exact frame limit conditions from main.py
            if ear < 0.25:
                self.eye_counter += 1
                if 10 <= self.eye_counter < 20:
                    driver_status = "DROWSY (WARNING)"
                    status_color = (0, 165, 255) 
                    local_alarm = "warning"
                elif self.eye_counter >= 20:
                    driver_status = "DROWSY (CRITICAL!)"
                    status_color = (0, 0, 255) 
                    local_alarm = "critical"
            else:
                self.eye_counter = 0

            if mar > 0.60:
                self.yawn_counter += 1
                if self.yawn_counter >= 15:
                    driver_status = "YAWNING DETECTED"
                    status_color = (0, 165, 255)
                    if local_alarm == "none":
                        local_alarm = "warning"
            else:
                self.yawn_counter = 0

            cv2.putText(img, f"STATUS: {driver_status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(img, f"EAR: {ear:.2f}", (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, f"MAR: {mar:.2f}", (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Communicate tracking state safely out to the main container
        if st.session_state.alarm_state != local_alarm:
            st.session_state.alarm_state = local_alarm
            
        return img

webrtc_streamer(
    key="drowsiness-detection", 
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Main web dashboard interface sound engine layer
if st.session_state.alarm_state == "critical":
    st.audio(DROWSY_ALARM_URL, autoplay=True, loop=True)
    st.error("🚨 CRITICAL SLEEP ATTACK DETECTED! SOUND ON!")
elif st.session_state.alarm_state == "warning":
    st.audio(YAWN_ALARM_URL, autoplay=True, loop=True)
    st.warning("⚠️ FATIGUE WARNING SIGNALS DETECTED!")
