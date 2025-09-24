# AD System: Face Detection & Gender Classification

An intelligent advertising system that uses advanced computer vision techniques to deliver targeted advertisements based on real-time face detection and gender classification. The system automatically plays relevant ads based on the detected audience demographics.

## 🎯 Key Features

- **Real-Time Face Detection**: Uses YOLOv8 for accurate and fast face detection
- **Gender Classification**: TensorFlow/Keras CNN model classifies detected faces as male/female
- **Smart Ad Targeting**: Automatically plays targeted ads based on audience demographics:
  - Male-only audience → Men's advertisements
  - Female-only audience → Women's advertisements
  - Mixed audience → Family advertisements
- **Individual Tracking**: Centroid-based tracking system tracks unique individuals over time
- **Modern UI**: Tkinter-based management interface with ttkbootstrap styling
- **Data Analytics**: Sends real-time detection data to remote API for analytics
- **Robust Performance**: Optimized for real-time processing with error handling

## 🏗️ Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam/Camera │───▶│  Face Detection  │───▶│ Gender Analysis │
│                 │    │   (YOLOv8)       │    │   (TensorFlow)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Centroid      │    │   Ad Selection   │    │   Data Logging  │
│   Tracking      │    │   & Playback     │    │   (API)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **YOLOv8 Model** (`Models/best10.pt`): Custom-trained face detection model
- **CNN Model** (`Models/Gender_Lastv3_last.h5`): TensorFlow gender classification model
- **Centroid Tracker**: Tracks individual faces across video frames
- **MediaPipe Integration**: Enhanced facial landmark detection
- **RESTful API**: Sends detection data to remote analytics server

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Minimum 4GB RAM (8GB recommended for optimal performance)
- NVIDIA GPU (optional, for hardware acceleration)

## 🚀 Installation

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd AD-System_using_Face_Detection_and_Gender_classifiction_using_computer_vision_
python -m venv ad_system_env
source ad_system_env/bin/activate  # On Windows: ad_system_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import cv2, tensorflow, ultralytics, mediapipe; print('All dependencies installed successfully')"
```

## 🎮 Usage

### Basic Operation

1. **Start the System**:
   ```bash
   python main.py
   ```

2. **Using the UI**:
   ```bash
   python ui.py
   ```

### Command Line Options

- Press `q` to quit the application
- The system will automatically detect faces and play appropriate ads
- Detection data is sent to the configured API endpoint

### UI Controls

The graphical interface provides:
- **Start System**: Begins face detection and ad playback
- **Stop System**: Halts all operations
- **Open Dashboard**: Opens web-based analytics dashboard
- **Upload New Ad**: Add new advertisement videos
- **Remove Ad**: Delete existing advertisements

## ⚙️ Configuration

### API Configuration

Edit the following in `main.py`:
```python
endpoint_url = "https://ads-track.smaster.live/api.php"
key = "kOEjOeaoL7BmgxC6PCM5GZsetaxq698hzgHv81Kd6XxfTsOM2W"
```

### Video Paths

Update advertisement paths in `main.py`:
```python
men_ad_path = 'AD_Videos/MenAD.mp4'
women_ad_path = 'AD_Videos/womanAD.mp4'
family_ad_path = 'AD_Videos/Familyad.mp4'
```

### Detection Parameters

Modify in `main.py`:
- **Timer thresholds**: Adjust `gender_timer` thresholds (default: 2 seconds)
- **Model confidence**: Modify detection confidence in `yolo_face_detection.py`
- **Tracking parameters**: Adjust `CentroidTracker` parameters

## 📊 Data Format

The system sends JSON data to the API endpoint:

```json
{
  "key": "kOEjOeaoL7BmgxC6PCM5GZsetaxq698hzgHv81Kd6XxfTsOM2W",
  "gender": "male|female|both",
  "datetime": "2024-01-15 14:30:25"
}
```

## 🔧 Troubleshooting

### Common Issues

**1. Webcam Not Detected**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

**2. Model Loading Errors**
- Ensure model files exist in `Models/` directory
- Check file permissions and paths
- Verify TensorFlow and OpenCV versions

**3. GPU Issues (if using CUDA)**
```bash
# Check CUDA availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**4. API Connection Errors**
- Verify internet connection
- Check API endpoint URL and authentication key
- Confirm server is running and accessible

**5. Performance Issues**
- Close unnecessary applications
- Ensure adequate RAM (minimum 4GB)
- Consider reducing video resolution if needed
- Use GPU acceleration if available

### Debug Mode

Run individual components for testing:
```bash
# Test face detection
python yolo_face_detection.py

# Test gender classification
python gender_classification.py

# Test tracking
python centroid_tracker.py
```

## 📁 Project Structure

```
AD-System_using_Face_Detection_and_Gender_classifiction_using_computer_vision_/
├── AD_Videos/                 # Advertisement video files
│   ├── Familyad.mp4          # Family advertisement
│   ├── MenAD.mp4             # Male-targeted advertisement
│   └── womanAD.mp4           # Female-targeted advertisement
├── Models/                   # Machine learning models
│   ├── best10.pt             # YOLOv8 face detection model
│   └── Gender_Lastv3_last.h5 # TensorFlow gender classification model
├── centroid_tracker.py       # Individual face tracking system
├── gender_classification.py  # Gender classification logic
├── load_model.py            # Model loading utilities
├── main.py                  # Core application logic
├── ui.py                    # Graphical user interface
├── yolo_face_detection.py   # YOLOv8 face detection
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for face detection
- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision utilities
- MediaPipe for facial landmark detection
- ttkbootstrap for UI styling

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information

---

**Note**: This system processes video data in real-time and sends analytics data to external servers. Ensure compliance with local privacy regulations when deploying in production environments.
