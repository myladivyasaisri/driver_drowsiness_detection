# 🚗 Vision-Based Advanced Driver Assistance System (ADAS) for Real-Time Fatigue & Anomaly Detection

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://opencv.org)
[![Target Hardware](https://shields.io)](https://raspberrypi.com)
[![Operational Status](https://shields.io)]()

An enterprise-grade computer vision solution deployed to mitigate commercial and transit accidents caused by operator exhaustion and micro-sleep events [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System]. This framework leverages localized multi-node facial mesh mapping to capture high-frequency physical anomalies (eyelid drooping and yawning vectors), executing deterministic tiered safety countermeasures in real-time.

---

## ⚙️ Core System Architecture Features

| Sub-System Component | Engineering Specification | Implementation Advantage |
| :--- | :--- | :--- |
| **Unified Chrome-Green Mesh** | 2D Point Contours Mapping (`BGR: 0, 255, 0`) | Provides high-contrast isolation overlays around ocular and labial zones during low-light driver environments [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System]. |
| **Tiered Alarm Escalation** | Multi-Stage Signal Evaluation Engine | Bypasses binary alert limits. Differentiates between soft fatigue (warning) and critical micro-sleep states (alarm). |
| **Threaded Audio Execution** | Asynchronous Background Subprocesses (`threading`) | Prevents main thread pipeline drops; keeps camera loop processing stable at a constant frame rate without lag [AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System]. |
| **Automotive Black Box Log** | Telemetry Stream Appender (`drowsiness_logs.txt`) | Generates chronological, immutable data footprints with calendar timestamps for corporate audit compliance. |

---

## 🔬 Mathematical Framework & Diagnostics Telemetry

The mathematical backbone utilizes scaling-invariant spatial ratios. This guarantees execution parity regardless of camera distance variations from the dashboard.

### 1. Eye Aspect Ratio (EAR) Matrix
Tracks 6 localized spatial coordinates surrounding the eyelids structure to isolate micro-sleep signatures:

\[\text{EAR} = \frac{\vert{}\vert{}p_2 - p_6\vert{}\vert{} + \vert{}\vert{}p_3 - p_5\vert{}\vert{}}{2 \vert{}\vert{}p_1 - p_4\vert{}\vert{}}\]

*   **`🟢 STATUS: AWAKE`** → EAR ≥ 0.25 (Stable tracking behavior)
*   **`🟠 STATUS: DROWSY (WARNING)`** → EAR < 0.25 (≥ 10 continuous frames; triggers soft chime loop)
*   **`🔴 STATUS: DROWSY (CRITICAL!)`** → EAR < 0.25 (≥ 20 continuous frames; executes high-pitched rapid double-beep sirens)

### 2. Mouth Aspect Ratio (MAR) Matrix
Tracks 20 internal labial nodes to continuously compute vertical expansion vectors during active driving cycles:

\[\text{MAR} = \frac{\vert{}\vert{}p_{51} - p_{59}\vert{}\vert{} + \vert{}\vert{}p_{53} - p_{57}\vert{}\vert{}}{2 \vert{}\vert{}p_{49} - p_{55}\vert{}\vert{}}\]

*   **Normal Operations** → MAR ≤ 0.60 (Speaking/breathing baseline)
*   **`🟠 STATUS: YAWNING DETECTED`** → MAR > 0.60 (≥ 15 continuous frames; triggers fatigue cooldown routines)

---

## 📁 System Directory Blueprint

```text
driver_drowsiness_detection/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # Compiled network weights database
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                             # Telemetry calculation formulas (EAR, MAR)
│
├── config.py                                  # Modular operational thresholds hub
├── main.py                                    # Execution engine & high-FPS video loop
├── requirements.txt                           # Dependency installer document
└── drowsiness_logs.txt                        # Automated diagnostic history document
```

---

## 📊 Black Box Telemetry Sheet Sample (`drowsiness_logs.txt`)
The framework writes atomic operation entries automatically into a local repository file for administrative reviews:
```text
[2026-07-21 14:30:05] SYSTEM START: Safety monitoring initialized.
[2026-07-21 14:32:14] ALERT: Driver was Drowsy - Duration: 2.45 seconds
[2026-07-21 14:35:40] NOTICE: Yawn Pattern Logged
[2026-07-21 14:40:12] SYSTEM SHUTDOWN: Monitoring ended cleanly.
```

---

## 💻 Installation & Pipeline Execution

### 1. Environment Package Dependencies Configuration
Deploy all mandatory dependencies into your localized sandbox compiler terminal environment:
```bash
pip install -r requirements.txt
```

### 2. Facial Landmark Weight Map Fetch
Ensure your 68-point dlib neural weights binary file is situated inside the specific directory target:
1. Fetch the binary asset from the [Official Dlib Repository Container](https://huggingface.co).
2. Save the `.dat` file inside the `models/` directory.

### 3. Initialize Engine Stream
Fire up the central driver safety cockpit application frame loop via PowerShell or Command Prompt:
```bash
python main.py
```

---

## 🎮 Interface Controls
*   **Key Command `q`**: Safely terminates camera sensor input pipelines, flushes stream buffer allocations to the logging file, silences background thread signals, and closes active GUI display frames cleanly.

---

## 👩‍💻 Principal Engineer & Author

### MYLA DIVYA SAI SRI
*Artificial Intelligence & Machine Learning (AI/ML) Student*

- 💼 **[Connect on LinkedIn](www.linkedin.com/in/myladivyasaisri)**
- 🐙 **[Follow on GitHub](https://github.com/myladivyasaisri)**
