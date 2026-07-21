import cv2
import dlib
import imutils
import sys
import os
import time
import winsound  
import threading 
from datetime import datetime
from imutils import face_utils

import config
from utils.metrics import calculate_ear, calculate_mar

STOP_ALARM_EVENT = threading.Event()
LOG_FILE = "drowsiness_logs.txt"

def log_event(event_type, duration=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        if duration:
            f.write(f"[{timestamp}] {event_type} - Duration: {duration:.2f} seconds\n")
        else:
            f.write(f"[{timestamp}] {event_type}\n")

def play_escalated_alarm(intensity):
    start_time = time.time()
    while not STOP_ALARM_EVENT.is_set() and (time.time() - start_time) < 4.0:
        if intensity == "critical":
            winsound.Beep(2800, 300)
            time.sleep(0.05)
            winsound.Beep(2800, 300)
        else:
            winsound.Beep(2000, 400)
        time.sleep(0.1)

def main():
    print("[DEBUG 1] Advanced Automotive Framework Active.")
    log_event("SYSTEM START: Safety monitoring initialized.")
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Landmark file missing: {config.MODEL_PATH}")
        sys.exit(1)

    EYE_COUNTER = 0
    YAWN_COUNTER = 0
    ALARM_ACTIVE = False
    DROWSE_START_TIME = None
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.MODEL_PATH)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    print("[DEBUG 4] Bypassing Windows architecture locks via DirectShow...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- విండోను పెద్దగా మార్చే ప్రొఫెషనల్ సెటప్ ---
    window_name = "ADAS Infrastructure Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # విండోను రీసైజ్ చేయడానికి అనుమతిస్తుంది
    cv2.resizeWindow(window_name, 800, 600)          # విండోను పెద్ద సైజుగా (800x600) సెట్ చేస్తుంది

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # గమనిక: స్పీడ్ కోసం ఇమేజ్ ప్రాసెసింగ్ 450px లోనే జరుగుతుంది, కానీ డిస్ప్లే పెద్దగా వస్తుంది
        frame = imutils.resize(frame, width=config.FRAME_WIDTH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        driver_status = "AWAKE"
        status_color = (0, 255, 0) 
        alarm_intensity = None

        if len(rects) == 0:
            cv2.putText(frame, "STATUS: SCANNING ROAD", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
            mar = calculate_mar(mouth)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)

            if ear < config.EAR_THRESHOLD:
                EYE_COUNTER += 1
                if 10 <= EYE_COUNTER < config.CONSECUTIVE_FRAMES_LIMIT:
                    driver_status = "DROWSY (WARNING)"
                    status_color = (0, 165, 255) 
                    alarm_intensity = "warning"
                    if DROWSE_START_TIME is None:
                        DROWSE_START_TIME = time.time()
                elif EYE_COUNTER >= config.CONSECUTIVE_FRAMES_LIMIT:
                    driver_status = "DROWSY (CRITICAL!)"
                    status_color = (0, 0, 255) 
                    alarm_intensity = "critical"
            else:
                if EYE_COUNTER >= 10 and DROWSE_START_TIME:
                    drowse_duration = time.time() - DROWSE_START_TIME
                    log_event("ALERT: Driver was Drowsy", drowse_duration)
                EYE_COUNTER = 0
                DROWSE_START_TIME = None

            if mar > config.MAR_THRESHOLD:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= config.YAWN_FRAMES_LIMIT:
                    driver_status = "YAWNING DETECTED"
                    status_color = (0, 165, 255)
                    if alarm_intensity is None:
                        alarm_intensity = "warning"
            else:
                if YAWN_COUNTER >= config.YAWN_FRAMES_LIMIT:
                    log_event("NOTICE: Yawn Pattern Logged")
                YAWN_COUNTER = 0

            if alarm_intensity is not None:
                if not ALARM_ACTIVE:
                    ALARM_ACTIVE = True
                    STOP_ALARM_EVENT.clear()
                    t = threading.Thread(target=play_escalated_alarm, args=(alarm_intensity,))
                    t.daemon = True
                    t.start()
            else:
                if ALARM_ACTIVE:
                    ALARM_ACTIVE = False
                    STOP_ALARM_EVENT.set()

            # UI Text elements positioning tuning
            cv2.putText(frame, f"STATUS: {driver_status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (340, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (340, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # షో విండో ఎలిమెంట్స్
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            STOP_ALARM_EVENT.set()
            log_event("SYSTEM SHUTDOWN: Monitoring ended cleanly.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
