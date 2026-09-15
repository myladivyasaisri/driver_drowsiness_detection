import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import dlib
import numpy as np

st.set_page_config(page_title="ADAS Infrastructure Dashboard", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("An Edge-AI Safety System for Real-Time Fatigue Monitoring")

# 1. Inject Hidden Browser Audio Engine (Bypasses winsound restrictions)
st.markdown(
    """
    <audio id="warning_alarm" loop>
        <source src="https://soundjay.com" type="audio/mp3">
    </audio>
    <audio id="critical_alarm" loop>
        <source src="https://soundjay.com" type="audio/mp3">
    </audio>
    <script>
        function playBrowserAlarm(intensity) {
            var warn = document.getElementById('warning_alarm');
            var crit = document.getElementById('critical_alarm');
            if (intensity === 'critical') {
                crit.play().catch(e => {});
                warn.pause();
            } else if (intensity === 'warning') {
                warn.play().catch(e => {});
                crit.pause();
            } else {
                warn.pause();
                crit.pause();
            }
        }
    </script>
    """,
    unsafe_allow_html=True
)

# 2. Math Formulas from your utils/metrics
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

# Initialize Dlib Engine
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("Landmark file missing inside models/ folder!")

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
        alarm_intensity = "none"

        if len(rects) == 0:
            cv2.putText(img, "STATUS: SCANNING ROAD", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        for rect in rects:
            shape = predictor(gray, rect)
            points = []
            for n in range(0, 68):
                points.append((shape.part(n).x, shape.part(n).y))
            
            # Match your main.py array slicing logic
            leftEye = points[36:42]
            rightEye = points[42:48]
            mouth = points[48:68]

            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
            mar = calculate_mar(mouth)

            # Draw your original project mesh outlines
            cv2.polylines(img, [np.array(leftEye)], True, (0, 255, 0), 1)
            cv2.polylines(img, [np.array(rightEye)], True, (0, 255, 0), 1)
            cv2.polylines(img, [np.array(mouth)], True, (0, 255, 0), 1)

            # Your exact main.py frame logic counters
            if ear < 0.25: # config.EAR_THRESHOLD
                self.eye_counter += 1
                if 10 <= self.eye_counter < 20: # config.CONSECUTIVE_FRAMES_LIMIT
                    driver_status = "DROWSY (WARNING)"
                    status_color = (0, 165, 255) 
                    alarm_intensity = "warning"
                elif self.eye_counter >= 20:
                    driver_status = "DROWSY (CRITICAL!)"
                    status_color = (0, 0, 255) 
                    alarm_intensity = "critical"
            else:
                self.eye_counter = 0

            if mar > 0.60: # config.MAR_THRESHOLD
                self.yawn_counter += 1
                if self.yawn_counter >= 15: # config.YAWN_FRAMES_LIMIT
                    driver_status = "YAWNING DETECTED"
                    status_color = (0, 165, 255)
                    if alarm_intensity == "none":
                        alarm_intensity = "warning"
            else:
                self.yawn_counter = 0

            # Render exact Dashboard UI layout
            cv2.putText(img, f"STATUS: {driver_status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(img, f"EAR: {ear:.2f}", (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, f"MAR: {mar:.2f}", (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Trigger client-side JavaScript sound mapping based on intensity status
        st.components.v1.html(f"<script>window.parent.playBrowserAlarm('{alarm_intensity}');</script>", height=0, width=0)
        return img

webrtc_streamer(
    key="drowsiness-detection", 
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
