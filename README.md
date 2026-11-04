# SafeSight
### Key tasks performed by the project

- **Detects** people, PPE items (hardhats, masks, safety vests), and machinery in images and video using the trained YOLO26 model.
- **Tracks people** across video frames so the system can follow their movement.
- **Checks zones** configured on the camera view and records when tracked people enter or remain in them.
- **Assesses PPE** for tracked people and flags detected missing PPE, with uncertain detections treated cautiously.
- **Monitors after-hours activity** using the configured site closing time and raises an alert when a person is detected after closing.
- **Displays a live dashboard** with camera feed, detection counts, zone occupancy, PPE alerts, machinery detections, and recent events.
- **Records events and evidence**, including timestamps, track IDs, event details, and saved image evidence for review.

In short, it turns camera detections into a basic site-safety monitoring and event-review workflow.

# AI-Based PPE Detection and Safety Compliance Monitoring System using YOLO26

An intelligent computer vision-based safety monitoring system designed to detect Personal Protective Equipment (PPE) violations among workers in construction environments using **YOLO26, OpenCV, and Deep Learning**.

The system identifies workers, detects required safety equipment, monitors PPE compliance, and highlights violations to support workplace safety management.

## Overview

Personal Protective Equipment plays a critical role in preventing workplace injuries, particularly in construction environments.

Traditional safety monitoring depends heavily on manual supervision, which can be time-consuming and inconsistent.

This project introduces an automated PPE detection and monitoring solution using YOLOv11 to identify safety equipment and detect missing PPE in images and video streams.

## Key Features

- **YOLO26 Detection:** Deep learning-based object detection for PPE identification.
- **Worker Detection:** Identifies workers in monitored environments.
- **PPE Compliance Monitoring:** Checks whether workers have the required protective equipment.
- **Violation Detection:** Identifies missing or improperly equipped PPE.
- **Object Tracking:** Uses ByteTrack to maintain worker tracking across video frames.
- **Video Processing:** Upload and analyze recorded videos.
- **Live Camera Monitoring:** Supports real-time PPE detection through a camera.
- **Violation Alerts:** Highlights detected safety violations.
- **Snapshot Capture:** Saves images of detected violations.
- **Monitoring Dashboard:** Displays safety monitoring statistics.
- **CSV Reporting:** Supports exporting violation records for analysis.

## PPE Categories

The detection model includes classes related to:

| Category | Description |
|---|---|
| Helmet | Head protection |
| Gloves | Hand protection |
| Safety Vest | High-visibility clothing |
| Safety Boots | Foot protection |
| Glasses | Eye protection |
| Human | Worker detection |
| No Helmet / No Hat | Head protection violation |
| No Gloves | Hand protection violation |
| No Vest | Safety vest violation |
| No Boots | Foot protection violation |

*The exact classes depend on the trained dataset and model configuration.*

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO26 | Object detection |
| Ultralytics | YOLO model framework |
| OpenCV | Image and video processing |
| PyTorch | Deep learning framework |
| ByteTrack | Object tracking |
| Streamlit | Interactive web application |
| Pandas | Violation data processing |
| NumPy | Numerical operations |
| CSV | Report generation |

## System Workflow

1. **Input Acquisition:** Upload a video or access a live camera feed.
2. **Frame Processing:** Extract frames using OpenCV.
3. **Object Detection:** YOLO26 identifies workers and PPE objects.
4. **Object Tracking:** ByteTrack maintains object identities across frames.
5. **Compliance Analysis:** Evaluate the detected PPE against required safety equipment.
6. **Violation Identification:** Identify missing protective equipment.
7. **Alert Generation:** Highlight workers with detected violations.
8. **Data Recording:** Store violation information and snapshots.
9. **Report Generation:** Generate CSV-based violation reports.

## Project Structure

```text
New_PPE_detection_YOLO/
│
├── models/
│   └── trained/
│       └── best.pt
│
├── datasets/
│   └── construction-ppe/
│
├── runs/
│
├── app.py
├── main.py
├── requirements.txt
│
├── data.yaml
├── bytetrack.yaml
│
├── snapshots/
├── reports/
│
└── README.md
```

*The structure above is illustrative. Adjust it to match the actual repository files.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Jeevan5125/New_PPE_detection_YOLO.git
```

### 2. Navigate to the Project

```bash
cd New_PPE_detection_YOLO
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If dependencies are not fully listed in `requirements.txt`, install the required packages:

```bash
pip install ultralytics opencv-python streamlit torch pandas numpy
```

### 5. Verify YOLO Installation

```bash
yolo checks
```

### 6. Run the Application

If your main Streamlit entry point is `app.py`:

```bash
streamlit run app.py
```

Use the actual application entry-point filename if it differs.

## Model Information

| Component | Details |
|---|---|
| Detection Architecture | YOLO26 |
| Model Variant | YOLOv11 Nano |
| Framework | Ultralytics |
| Tracking Algorithm | ByteTrack |
| Trained Model | `models/trained/best.pt` |
| Dataset | Construction PPE Dataset |

The trained model is used to identify workers and PPE-related objects in video frames.

## Performance and Monitoring

The application focuses on monitoring workplace safety through:

- Worker detection and tracking.
- PPE compliance classification.
- Missing equipment identification.
- Violation snapshots.
- Detection logs.
- Safety violation statistics.
- CSV-based reporting.

Actual model performance metrics should be added after evaluating the trained model on a separate test dataset.

## Potential Future Enhancements

- Multi-camera safety monitoring.
- Advanced worker identification and tracking.
- PPE violation severity classification.
- Automated email or notification alerts.
- Safety compliance trend analysis.
- Explainable AI for detection decisions.
- Cloud-based centralized monitoring.
- Advanced analytics for workplace safety management.

## Applications

- Construction sites.
- Industrial manufacturing facilities.
- Warehouses.
- Infrastructure development projects.
- Occupational health and safety monitoring.

## Disclaimer

This project is developed for academic and research purposes. Automated PPE detection may produce false positives and false negatives. It should be used as an assistive monitoring solution rather than a replacement for human safety inspections.

## Author

**Jeevan Rajkumar**

Integrated M.Tech – Software Engineering  
VIT Vellore

GitHub: [Jeevan5125](https://github.com/Jeevan5125)

---

⭐ If you find this project useful, consider starring the repository!
