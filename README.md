# Real-Time Driver Drowsiness and Yawn Detection System

A computer vision-based Advanced Driver Assistance System (ADAS) designed to prevent road accidents by tracking driver fatigue in real-time. The system processes live video streams to compute the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), automatically triggering a tiered safety alarm when drowsiness or yawning patterns are detected [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System].

---

## 🚀 Features

- **Drowsiness Tracking (EAR)**: Monitors 6 distinct facial landmarks per eye to track micro-sleeps and prolonged eye closures.
- **Yawn Detection (MAR)**: Analyzes 20 inner and outer lip landmark nodes to recognize repetitive yawning patterns.
- **Threaded Audio Alert System**: Triggers an automated audio alarm in an isolated background thread to prevent display lag or video freezing.
- **Automotive Safety Control (3-5 Sec Warning Loop)**: Follows real-world vehicle warning standards, sustaining a professional 4-second safety buzzer wave to alert the driver before automatically resetting [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System].
- **Windows Device Integration**: Features an optimized DirectShow (`CAP_DSHOW`) camera initialization sequence to bypass common Windows hardware locking issues.

---

## 📁 Project Structure

```text
driver_drowsiness_detection/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # Pre-trained dlib coordinates model
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                             # Telemetry calculation formulas (EAR, MAR)
│
├── config.py                                  # Modular sensitivity thresholds hub
├── main.py                                    # Execution engine & camera processing loop
└── requirements.txt                           # Dependency installer document
```

---

## 🛠️ Mathematical Foundations

### 1. Eye Aspect Ratio (EAR)
The system tracks 6 landmark points around each eye to monitor eyelids spacing:
$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$
*   **Normal State**: `0.25 - 0.30`
*   **Drowsy State**: Drops below `0.25` for more than 20 consecutive frames.

### 2. Mouth Aspect Ratio (MAR)
The system calculates the spatial layout of inner lip points to track mouth openness:
$$\text{MAR} = \frac{||p_{51} - p_{59}|| + ||p_{53} - p_{57}||}{2 ||p_{49} - p_{55}||}$$
*   **Normal State**: `0.10 - 0.30`
*   **Yawning State**: Spikes above `0.60` for more than 15 consecutive frames.

---

## 💻 Setup & Installation Instructions

### 1. Environment Setup
Clone or open your project directory and run the following command to install all mandatory framework modules:
```bash
pip install -r requirements.txt
```

### 2. Download the Pre-trained Weights Model
Dlib requires the 68-point shape predictor map file to pinpoint facial telemetry:
1. Download the file from the [Official Dlib Model Repository](https://huggingface.co).
2. Move the downloaded `shape_predictor_68_face_landmarks.dat` file directly inside your `models/` directory.

### 3. Run the Application
Boot up the main project frame via PowerShell or Command Prompt:
```bash
python main.py
```

---

## 🎯 Usage Keybindings
*   **Press `q`**: Safely terminates the camera stream loop, silences active alerts, and shuts down all UI frames.

---

## 👩‍💻 Author
- **MYLA DIVYA SAI SRI**
