import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import dlib
import numpy as np
import av

st.set_page_config(page_title="ADAS Dashboard", layout="wide")
st.title("Driver Drowsiness Detection System")
st.subheader("Real-Time Fatigue Monitoring Interface")

st.markdown(
    """
    <audio id="drowsy_siren" loop>
        <source src="https://soundjay.com" type="audio/mp3">
    </audio>
    <audio id="yawn_siren" loop>
        <source src="https://soundjay.com" type="audio/mp3">
    </audio>
    <script>
        function manageAlerts(status) {
            var drowsyAudio = window.parent.document.getElementById('drowsy_siren');
            var yawnAudio = window.parent.document.getElementById('yawn_siren');
            if (!drowsyAudio || !yawnAudio) return;
            
            if (status === 'DROWSY') {
                drowsyAudio.play().catch(function(e) {});
                yawnAudio.pause();
            } else if (status === 'YAWNING') {
                yawnAudio.play().catch(function(e) {});
                drowsyAudio.pause();
            } else {
                drowsyAudio.pause();
                yawnAudio.pause();
            }
        }
    </script>
    """,
    unsafe_allow_html=True
)

def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = np.linalg.norm(np.array(mouth[13]) - np.array(mouth[19]))
    B = np.linalg.norm(np.array(mouth[15]) - np.array(mouth[17]))
    C = np.linalg.norm(np.array(mouth) - np.array(mouth[6]))
    return (A + B) / (2.0 * C)

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except:
    st.error("Missing model data file inside models/ folder!")

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.eye_counter = 0
        self.yawn_counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        status = "AWAKE"
        color = (0, 255, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            points = []
            for n in range(0, 68):
                points.append((shape.part(n).x, shape.part(n).y))
                cv2.circle(img, (shape.part(n).x, shape.part(n).y), 1, (0, 255, 0), -1)
            
            leftEye = points[36:42]
            rightEye = points[42:48]
            mouth = points[48:68]

            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
            mar = calculate_mar(mouth)

            if ear < 0.25:
                self.eye_counter += 1
                if self.eye_counter >= 15:
                    status = "DROWSY"
                    color = (0, 0, 255)
            else:
                self.eye_counter = 0

            if mar > 0.60:
                self.yawn_counter += 1
                if self.yawn_counter >= 15:
                    if status != "DROWSY":
                        status = "YAWNING"
                        color = (0, 165, 255)
            else:
                self.yawn_counter = 0

            cv2.putText(img, f"STATUS: {status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, f"EAR: {ear:.2f} MAR: {mar:.2f}", (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        st.components.v1.html(f"<script>window.parent.manageAlerts('{status}');</script>", height=0, width=0)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
