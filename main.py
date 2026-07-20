import cv2
import dlib
import imutils
import sys
import os
import time
import winsound  
import threading 
from imutils import face_utils

import config
from utils.metrics import calculate_ear, calculate_mar

# Global control flag to manage the audio thread state
STOP_ALARM_EVENT = threading.Event()

def play_sustained_alarm():
    """
    Rings a looping alarm for a professional automotive standard duration (4 seconds).
    Stops cleanly once the driver opens their eyes after the safety window expires.
    """
    start_time = time.time()
    
    # Run loop for exactly 4.0 seconds max to prevent driver audio distraction
    while not STOP_ALARM_EVENT.is_set() and (time.time() - start_time) < 4.0:
        if os.path.exists(config.ALARM_PATH):
            winsound.PlaySound(config.ALARM_PATH, winsound.SND_FILENAME)
        else:
            # Standard high-pitched vehicle alert chime (2500Hz for 500ms)
            winsound.Beep(2500, 500)
            
        time.sleep(0.1)

def main():
    print("[DEBUG 1] Script started successfully.")
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Landmark file not found at: {config.MODEL_PATH}")
        sys.exit(1)

    # Tracking counters and condition state flags
    EYE_COUNTER = 0
    YAWN_COUNTER = 0
    ALARM_ACTIVE = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.MODEL_PATH)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    print("[DEBUG 4] Initializing camera streams...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[DEBUG 5] Camera index 0 failed. Trying index 1 via DirectShow...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[DEBUG 6] DirectShow fallback triggered. Trying index 0 standard...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[CRITICAL ERROR] Windows hardware rejected all webcam indexes.")
        sys.exit(1)
        
    # Set operational compression overrides to maximize streaming FPS performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Camera streaming actively running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera failed to capture a frame. Exiting stream loop.")
            break

        frame = imutils.resize(frame, width=config.FRAME_WIDTH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        show_drowsy_alert = False
        show_yawn_alert = False

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = calculate_ear(leftEye)
            rightEAR = calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = calculate_mar(mouth)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # --- 1. DROWSINESS EVALUATION ---
            if ear < config.EAR_THRESHOLD:
                EYE_COUNTER += 1
                if EYE_COUNTER >= config.CONSECUTIVE_FRAMES_LIMIT:
                    show_drowsy_alert = True
            else:
                EYE_COUNTER = 0

            # --- 2. YAWN EVALUATION ---
            if mar > config.MAR_THRESHOLD:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= config.YAWN_FRAMES_LIMIT:
                    show_yawn_alert = True
            else:
                YAWN_COUNTER = 0

            # --- 3. ALARM CONTROLLER LOOP ---
            if show_drowsy_alert or show_yawn_alert:
                if not ALARM_ACTIVE:
                    ALARM_ACTIVE = True
                    STOP_ALARM_EVENT.clear()  
                    
                    # Spawn the isolated background audio thread
                    t = threading.Thread(target=play_sustained_alarm)
                    t.daemon = True
                    t.start()
            else:
                if ALARM_ACTIVE:
                    ALARM_ACTIVE = False
                    STOP_ALARM_EVENT.set()

            # UI Text elements rendering
            if show_drowsy_alert:
                cv2.putText(frame, "DROWSINESS ALERT!!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if show_yawn_alert:
                cv2.putText(frame, "YAWN DETECTED!!!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Data telemetry display
            cv2.putText(frame, f"EAR: {ear:.2f}", (320, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (320, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Drowsiness & Yawn Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            STOP_ALARM_EVENT.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application shut down securely.")

if __name__ == "__main__":
    main()
