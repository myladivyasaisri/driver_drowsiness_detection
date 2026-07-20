import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")  # Path for custom audio file

# Eye (Drowsiness) Thresholds
EAR_THRESHOLD = 0.25               # Eye Aspect Ratio limit
CONSECUTIVE_FRAMES_LIMIT = 20      # Number of frames before eye alarm triggers

# Mouth (Yawn) Thresholds
MAR_THRESHOLD = 0.60               # Mouth Aspect Ratio limit for yawning (0.60 is standard)
YAWN_FRAMES_LIMIT = 15             # Number of frames before yawn alarm triggers

# Visual Settings
FRAME_WIDTH = 450                  # Video width resizing
