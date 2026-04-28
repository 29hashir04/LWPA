# High-Performance Gait Recognition System with Firebase Integration

A real-time gait recognition application built with Streamlit, PyTorch, and Firebase, featuring high-performance camera processing, cloud authentication, and scalable data storage.

## Features

- **Real-time Gait Recognition**: Continuous live camera feed with pose detection
- **High-Performance Processing**: GPU acceleration, batch processing, and optimized inference
- **Firebase Authentication**: Secure cloud-based user authentication and management
- **Cloud Data Storage**: Firestore integration for scalable gait pattern storage
- **Gait Enrollment**: Multi-sequence gait data collection with configurable parameters
- **Data Management**: CRUD operations, batch comparisons, and analytics
- **User Profiles**: Comprehensive user management and statistics
- **Clean Interface**: Professional UI without visual clutter

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Webcam
- Firebase project (see setup instructions below)

## Firebase Setup

Before running the application, you need to set up Firebase:

1. **Create Firebase Project**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Enable Authentication (Email/Password)
   - Create Firestore Database

2. **Configure Authentication**
   - See `FIREBASE_SETUP.md` for detailed instructions
   - Set up Streamlit secrets or environment variables
   - Configure Firestore security rules

3. **Install Firebase Dependencies**
   ```bash
   pip install firebase-admin pyrebase4
   ```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Firebase configuration (see `FIREBASE_SETUP.md`)

## Usage

1. Start the application:
   ```bash
   streamlit run main.py
   ```

2. Access the application at `http://localhost:8501`

3. Create a Firebase account or login
4. Enroll gait patterns in the "Enroll Gait" tab
5. Use "High-Performance Recognition" for real-time identification
6. Manage your data in "My Data" tab
7. View your profile in "Profile" tab

## Architecture

- **Frontend**: Streamlit web interface
- **Authentication**: Firebase Authentication with secure token management
- **Data Storage**: Firebase Firestore for cloud storage with local backup
- **Models**: YOLOv8n-pose for pose detection, FastPoseGait for gait recognition
- **Cloud Integration**: Real-time sync across devices and users

## Performance Features

- GPU acceleration with CUDA
- Batch processing for improved throughput
- Threading for camera capture and processing
- Real-time streaming with minimal latency
- Data quality filtering (removes corrupted patterns)
- Firebase caching and offline support

## File Structure

```
Recognition System/
├── main.py                           # Main application
├── firebase_setup/                   # Firebase configuration and services
│   ├── firebase_config.py            # Firebase configuration
│   ├── firebase_auth_service.py      # Firebase authentication
│   ├── firebase_user_manager.py      # User and data management
│   ├── FIREBASE_SETUP.md            # Firebase setup guide
│   └── gaitrecognitionapp-47c41-firebase-adminsdk-fbsvc-47efac1c13.json
├── models/                          # Gait recognition models
│   ├── gait_model.py                # OptimizedFastPoseGaitModel
│   └── __init__.py
├── data/                            # Data management
│   ├── gait_data_manager.py         # OptimizedGaitDataManager
│   └── __init__.py
├── processing/                      # Camera processing
│   ├── camera_processor.py          # HighPerformanceCameraProcessor
│   └── __init__.py
├── ui/                             # User interface
│   ├── render_functions.py          # UI rendering functions
│   └── __init__.py
├── utils/                          # Utility functions
│   ├── data_analysis.py             # Data analysis and visualization
│   └── __init__.py
├── local_auth/                     # Local authentication fallback
│   ├── simple_auth_working.py      # Local authentication system
│   ├── simple_data_manager.py      # Local data management
│   └── __init__.py                 # Package initialization
├── requirements.txt                # Dependencies
├── yolov8n-pose.pt                # YOLO pose detection model
├── gait_data/                     # User gait data storage
│   ├── demo_gait.pkl
│   └── Hashir_gait.pkl
└── README.md                       # This file
```

## Firebase Integration Features

- **Secure Authentication**: Firebase Auth with email/password
- **Cloud Storage**: Firestore for gait patterns and user data
- **Real-time Sync**: Data synchronization across devices
- **User Management**: Profile management and statistics
- **Scalability**: Supports multiple users and large datasets
- **Backup System**: Local fallback for offline operation

## Technical Details

- **Pose Detection**: YOLOv8n-pose model
- **Gait Recognition**: FastPoseGait neural network
- **Feature Extraction**: 128-dimensional gait features
- **Recognition Threshold**: 50% confidence
- **Data Quality**: Filters patterns with norms > 1000
- **Cloud Storage**: Firestore with automatic backup to local files

## Migration from Local Auth

If you're upgrading from the local authentication system:

1. **Backup existing data**:
   - Export gait data from "My Data" tab
   - Save user_data.json file

2. **Set up Firebase**:
   - Follow `FIREBASE_SETUP.md` instructions
   - Create Firebase accounts for existing users

3. **Import data**:
   - Use the new Firebase-based enrollment system
   - Data will be automatically synced to the cloud

## Troubleshooting

- **Firebase Connection Issues**: Check your configuration in `FIREBASE_SETUP.md`
- **Authentication Errors**: Verify Firebase project settings and API keys
- **GPU Issues**: Ensure CUDA drivers are installed for acceleration
- **Webcam Issues**: Check webcam permissions and availability
- **Data Sync Issues**: Check Firestore security rules and permissions

## Security Features

- **Firebase Security Rules**: User-specific data access
- **Token Management**: Secure authentication tokens
- **Data Encryption**: Firestore automatic encryption
- **Access Control**: Role-based permissions

## License

This project is for educational and research purposes.