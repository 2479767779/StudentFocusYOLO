# Student Focus Monitoring System

## 📖 Project Overview

This project is a real-time classroom student focus monitoring system based on the YOLO (You Only Look Once) deep learning model. Utilizing computer vision technology, the system can automatically detect and analyze students' focus levels in the classroom, providing teachers with real-time feedback and data analysis.

### 🎯 Core Features
- **Real-time Face Detection**: Uses YOLOv8 to detect students' faces in the classroom.
- **Pose Estimation**: Analyzes focus indicators such as head pose and gaze direction.
- **Focus Scoring**: Calculates real-time focus scores based on multi-dimensional features.
- **Data Visualization**: Generates focus trend charts and statistical reports.
- **Anomaly Warning**: Promptly alerts teachers when a lack of concentration is detected.

## 🛠️ Technical Architecture

### Core Tech Stack
- **Deep Learning Framework**: PyTorch + Ultralytics YOLOv8
- **Computer Vision**: OpenCV, MediaPipe
- **Facial Feature Analysis**: Dlib, cv2.face
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit/FastAPI

### Algorithm Principle
```
Focus Evaluation Model = 
  Face Detection (YOLO) + 
  Pose Estimation (MediaPipe) + 
  Gaze Tracking (Dlib) + 
  Attention Scoring Algorithm
```

## 📦 Installation Requirements

### Environment Requirements
- Python 3.8+
- CUDA 11.8+ (Recommended, CPU mode optional)
- 4GB+ RAM (8GB+ recommended)
- Camera device

### Quick Installation

```bash
# 1. Clone the project
git clone https://github.com/yourusername/student-focus-yolo.git
cd student-focus-yolo

# 2. Create virtual environment
conda create -n focus_monitor python=3.9
conda activate focus_monitor

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained models
python scripts/download_models.py
```

### Dependency List
```txt
ultralytics==8.0.0
opencv-python==4.8.0
mediapipe==0.10.0
dlib==19.24.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.15.0
streamlit==1.25.0
fastapi==0.103.0
```

## 🚀 Quick Start

### 1. Basic Usage - Real-time Monitoring

```bash
# Start real-time camera monitoring
python main.py --source 0 --display --save-results

# Parameter Description
# --source 0: Use default camera (can specify video file path)
# --display: Display real-time feed
# --save-results: Save detection results
```

### 2. Batch Analysis - Video File Processing

```bash
# Analyze recorded classroom video
python scripts/evaluate.py --input videos/classroom_01.mp4 --output results/classroom_01_analysis.json
```

### 3. Web Interface - Interactive Monitoring

```bash
# Start Web interface
streamlit run app/dashboard.py

# Or use FastAPI
uvicorn app.api:app --reload
```

## 📁 Project Structure

```
StudentFocusYOLO/
├── 📁 configs/                    # Configuration files
│   ├── model_config.yaml         # Model parameters
│   ├── focus_scoring.yaml        # Focus scoring rules
│   └── camera_calibration.yaml   # Camera calibration
│
├── 📁 data/                      # Data directory
│   ├── raw/                      # Raw videos/images
│   ├── processed/                # Processed results
│   └── models/                   # Trained models
│
├── 📁 src/                       # Core source code
│   ├── detection/                # Detection module
│   │   ├── face_detector.py      # YOLO face detection
│   │   ├── pose_estimator.py     # Pose estimation
│   │   └── gaze_tracker.py       # Gaze tracking
│   │
│   ├── analysis/                 # Analysis module
│   │   ├── focus_analyzer.py     # Focus analysis
│   │   ├── behavior_classifier.py # Behavior classification
│   │   └── statistics.py         # Statistical analysis
│   │
│   ├── utils/                    # Utility functions
│   │   ├── video_processor.py    # Video processing
│   │   ├── visualization.py      # Visualization
│   │   └── logger.py             # Logging
│   │
│   └── models/                   # Model definitions
│       ├── yolov8_focus.py       # YOLO extended model
│       └── focus_scorer.py       # Focus scorer
│
├── 📁 app/                       # Application UI
│   ├── dashboard.py              # Streamlit dashboard
│   ├── api.py                    # FastAPI interface
│   └── static/                   # Static resources
│
├── 📁 scripts/                   # Script tools
│   ├── download_models.py        # Download pre-trained models
│   ├── train_custom_model.py     # Train custom model
│   └── evaluate.py               # Model evaluation
│
├── 📁 tests/                     # Test code
├── 📁 docs/                      # Documentation
├── 📁 results/                   # Result output
│
├── requirements.txt              # Dependencies
├── main.py                       # Main program entry
├── README.md                     # Project documentation
└── LICENSE                       # License
```

## 🔬 Core Algorithm Details

### 1. Focus Scoring Model

The focus score is calculated based on the following dimensions:

```python
Focus Score = w1 × Pose Score + w2 × Gaze Score + w3 × Expression Score + w4 × Temporal Score

Where:
- Pose Score: Head rotation angle, tilt degree (0-1)
- Gaze Score: Alignment between gaze direction and the podium (0-1)  
- Expression Score: Eye opening degree, yawning frequency (0-1)
- Temporal Score: Attention duration, fluctuation status (0-1)
- Weights: w1=0.3, w2=0.3, w3=0.2, w4=0.2
```

### 2. YOLO Detection Optimization

Model optimizations for classroom scenarios:
- **Data Augmentation**: Lighting changes, occlusions, multiple angles.
- **Anchor Box Adjustment**: Adapted for classroom distances and angles.
- **Post-processing**: Non-Maximum Suppression (NMS) + Identity association.

### 3. Pose Estimation Workflow

```
Video Frame → YOLO Face Detection → MediaPipe Landmarks → 
Head Pose Calculation → Gaze Direction Estimation → Focus Mapping
```

## 📊 Output Examples

### Real-time Monitoring Interface
```
[Camera Feed]
Student A: Focus 85% ✓
Student B: Focus 62% ⚠
Student C: Focus 91% ✓

Class Average: 79%
Distracted: 1 person
```

### Analysis Report
```json
{
  "session_id": "class_20241218_0900",
  "duration": "90 minutes",
  "focus_statistics": {
    "average_score": 78.5,
    "peak_focus": 85.2,
    "attention_drop": 3,
    "distracted_time": "12 minutes"
  },
  "individual_reports": [
    {
      "student_id": "A001",
      "focus_score": 85,
      "attention_span": "45 minutes",
      "distraction_events": 2
    }
  ]
}
```

## 🎯 Use Cases (Course Project)

### Teaching Applications
- ✅ **Real-time Classroom Monitoring**: Teachers can understand student states in real-time.
- ✅ **Teaching Effect Evaluation**: Analyze the attractiveness of different teaching methods.
- ✅ **Personalized Tutoring**: Identify students who need extra attention.
- ✅ **Classroom Discipline Management**: Automatically detect behaviors like zoning out or sleeping.

### Research Applications
- ✅ **Educational Psychology Research**: Relationship between attention and learning outcomes.
- ✅ **Instructional Design Optimization**: Analyze the engagement levels of course content.
- ✅ **Online Education Evaluation**: Monitor focus levels in remote learning.

## ⚙️ Advanced Configuration

### Camera Settings
```yaml
# configs/camera_calibration.yaml
camera:
  resolution: [1280, 720]
  fps: 30
  brightness: 1.0
  contrast: 1.1
  focus_distance: 2.0  # Meters
```

### Focus Thresholds
```yaml
# configs/focus_scoring.yaml
thresholds:
  excellent: 85    # Excellent
  good: 70         # Good
  average: 50      # Average
  poor: 30         # Poor
```

### Performance Optimization
```bash
# Use GPU acceleration
python main.py --device cuda --batch-size 4

# Lightweight mode (CPU)
python main.py --device cpu --model yolov8n --half
```

## 🔍 Troubleshooting

### Common Issues

**Q: Camera cannot start**
```bash
# Check camera permissions
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try another camera
python main.py --source 1
```

**Q: Model download failed**
```bash
# Manually download and place in data/models/
# YOLOv8n: https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
```

**Q: Running speed is slow**
```bash
# Lower resolution
python main.py --source 0 --img-size 640

# Use a smaller model
python main.py --model yolov8n --img-size 416
```

## 📈 Performance Metrics

| Model | Accuracy | FPS (GPU) | FPS (CPU) | Memory Usage |
|------|--------|-----------|-----------|----------|
| YOLOv8n | 85% | 60 | 15 | 500MB |
| YOLOv8s | 90% | 45 | 8 | 1.2GB |
| YOLOv8m | 93% | 25 | 3 | 2.8GB |

## 🤝 Contribution Guide

Contributions and suggestions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Excellent YOLO implementation
- [MediaPipe](https://github.com/google/mediapipe) - Real-time pose estimation
- [OpenCV](https://opencv.org/) - Fundamental computer vision library
- All contributors in the open-source community

## 📞 Contact

For questions or suggestions, please contact via:
- Issue submission: [GitHub Issues](https://github.com/2479767779/student-focus-yolo/issues)
- Email: 2479767779@qq.com

---

**Note**: This system is for educational and research purposes only. Please ensure compliance with relevant privacy laws and obtain necessary authorization and consent during use.

**Version**: v1.0.1
**Last Updated**: 2025-12-18
