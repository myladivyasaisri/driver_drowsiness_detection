# Driver Drowsiness Detection System
### *An Edge-AI Safety System for Real-Time Fatigue Monitoring*

---

## 📌 Project Overview
This project is an advanced real-time computer vision framework designed to mitigate highway transit accidents caused by operator exhaustion and micro-sleep events [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System]. 

By processing live camera frames, the system maps facial telemetry landmarks using spatial equations to calculate eye and mouth ratios, instantly executing audio alarms when safety limits are breached.

---

## 🚀 Key System Features

*   **Dynamic HUD Status Display**: Real-time driver conditions are flashed directly onto the video feed layout (`AWAKE`, `DROWSY (WARNING)`, or `DROWSY (CRITICAL!)`).
*   **Unified Chrome-Green Mesh**: Standardized bright green boundaries (`BGR: 0, 255, 0`) draw clean outlines around ocular and labial zones for optimal low-light tracking [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System].
*   **Multi-Stage Escalated Alarm**:
    *   *Soft Warning*: Sounds single tones during repetitive yawn loops.
    *   *Critical Siren*: Sounds aggressive, rapid double-beeps during deep micro-sleeps.
*   **Threaded Audio Subprocesses**: Uses background threads to play alert buzzers asynchronously, preventing display frame drops or screen lag [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System].
*   **Black Box Event Logging**: Automatically records event logs with calendar timestamps into a local repository file for administrative safety audits.

---

## 📁 Repository Directory Map

```text
driver_drowsiness_detection/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # Neural network weights database
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                             # Formulas file (EAR & MAR math)
│
├── config.py                                  # Calibration thresholds hub
├── main.py                                    # Central camera loops engine
├── requirements.txt                           # Software dependencies list
└── drowsiness_logs.txt                        # Automated telemetry data file
```

---

## 🔬 Mathematical Framework & Diagnostics

The core analytical loops rely on scaling-invariant mathematical fractions, guaranteeing accurate processing regardless of how close or far the driver sits from the lens.

### 1. Eye Aspect Ratio (EAR)
Calculates horizontal and vertical eyelids vectors:
\[\text{EAR} = \frac{\vert{}\vert{}p_2 - p_6\vert{}\vert{} + \vert{}\vert{}p_3 - p_5\vert{}\vert{}}{2 \vert{}\vert{}p_1 - p_4\vert{}\vert{}}\]

*   **`🟢 STATUS: AWAKE`** → EAR ≥ 0.25
*   **`🟠 STATUS: DROWSY (WARNING)`** → EAR < 0.25 *(Triggers soft loop after 10 frames)*
*   **`🔴 STATUS: DROWSY (CRITICAL!)`** → EAR < 0.25 *(Triggers double-beep sirens after 20 frames)*

### 2. Mouth Aspect Ratio (MAR)
Calculates inner lip dilation layouts during active driving shifts:
\[\text{MAR} = \frac{\vert{}\vert{}p_{51} - p_{59}\vert{}\vert{} + \vert{}\vert{}p_{53} - p_{57}\vert{}\vert{}}{2 \vert{}\vert{}p_{49} - p_{55}\vert{}\vert{}}\]

*   **Normal Baseline** → MAR ≤ 0.60
*   **`🟠 STATUS: YAWNING DETECTED`** → MAR > 0.60 *(Triggers cooldown logging after 15 frames)*

---

## 📊 Black Box Logs Sample (`drowsiness_logs.txt`)
The system appends clean chronological records automatically during active run states:
```text
[2026-07-21 14:30:05] SYSTEM START: Safety monitoring initialized.
[2026-07-21 14:32:14] ALERT: Driver was Drowsy - Duration: 2.45 seconds
[2026-07-21 14:35:40] NOTICE: Yawn Pattern Logged
[2026-07-21 14:40:12] SYSTEM SHUTDOWN: Monitoring ended cleanly.
```

---

## 💻 Technical Setup & Execution

### 1. Dependencies Configuration
Deploy all mandatory dependencies into your localized sandbox environment by running:
```bash
pip install -r requirements.txt
```

### 2. Neural Weights Initialization
1. Download the pre-trained data file from the [Official Dlib Container](https://huggingface.co).
2. Save the compiled file as `shape_predictor_68_face_landmarks.dat` inside your `models/` directory.

### 3. Running the Engine
Initialize the central driver dashboard panel loop using your terminal interface:
```bash
python main.py
```

---

## 🎮 Interface Controls
*   **Key Command `q`**: Safely stops camera input channels, flushes file writing streams, terminates background threads, and kills display windows cleanly.

---

---

## 👩‍💻 Author

### MYLA DIVYA SAI SRI
*Artificial Intelligence & Machine Learning (AI/ML) Student*

*   **LinkedIn**: www.linkedin.com/in/myladivyasaisr
*   **GitHub**: https://github.com/myladivyasaisri

---

