# 😊 Emotion Detection from Face — EmotionVision AI

A strong computer-vision project that detects human faces and estimates facial emotions using **DeepFace**, OpenCV and Streamlit.

## Features

- Face detection
- 7 common emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Upload-image analysis
- Camera capture analysis
- Multiple-face detection
- Emotion confidence percentages
- Face bounding boxes
- Dominant emotion
- Emotion frequency chart
- Emotion distribution chart
- Detection history stored in CSV
- Downloadable history
- Clear-history control
- Selectable face detector
- Clean dashboard UI
- Automatic AI model download on first run

## Recommended environment

For the smoothest setup use:

- Windows 10/11
- Python **3.10 or 3.11**
- 8 GB RAM or more
- Internet connection for first-time DeepFace model downloads

> Python 3.14 is not recommended for this project because TensorFlow/DeepFace compatibility can lag behind new Python releases.

## Project structure

```text
Emotion_Detection_From_Face/
│
├── app.py
├── emotion_engine.py
├── requirements.txt
├── README.md
├── run.bat
├── setup.bat
│
├── assets/
│   └── README.txt
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── README.txt
│
└── .streamlit/
    └── config.toml
```

## Installation — Windows PowerShell

### 1. Open the project folder

```powershell
cd Emotion_Detection_From_Face
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv venv
```

If `py -3.11` is unavailable:

```powershell
python -m venv venv
```

### 3. Activate

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

### 6. Run

```powershell
streamlit run app.py
```

Then open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

## First run

DeepFace downloads required model files the first time it performs analysis. This can take several minutes depending on your internet speed.

Do not close the application while the model is being initialized.

## How to use

### Upload mode

1. Select **Upload Image**.
2. Upload JPG/PNG/WEBP.
3. Wait for analysis.
4. The app detects faces.
5. Each face receives a dominant emotion and confidence score.
6. Results can be saved to `data/emotion_history.csv`.

### Camera mode

1. Select **Camera**.
2. Click **Take a photo**.
3. Allow camera access.
4. Capture the frame.
5. The detected emotions appear below the image.

## Detector choices

### OpenCV
Fast and suitable for normal laptops.

### RetinaFace
Usually stronger face detection but significantly heavier/slower.

### MTCNN
Alternative detector for face localization.

Start with OpenCV.

## Troubleshooting

### TensorFlow / DeepFace installation problem

Use Python 3.10 or 3.11.

Check:

```powershell
python --version
```

Recommended:

```text
Python 3.11.x
```

### Port already in use

Run:

```powershell
streamlit run app.py --server.port 8502
```

### Camera does not appear

- Allow browser camera permission.
- Close applications currently using the camera.
- Restart Streamlit.
- Try Chrome or Edge.

### No face detected

Use a clear, front-facing image with good lighting. Lower the minimum confidence filter to 0%.

### Model download error

Make sure internet access is available and rerun:

```powershell
streamlit run app.py
```

## Resume description

**EmotionVision AI — Facial Emotion Detection System**

Developed a Python-based facial emotion recognition application using DeepFace, OpenCV and Streamlit. Implemented multi-face detection, emotion classification across seven classes, confidence scoring, image/camera analysis, interactive analytics and persistent CSV-based detection history. Designed a modular inference engine and user-friendly dashboard for real-time computer-vision experimentation.

## Suggested resume technologies

Python • OpenCV • DeepFace • TensorFlow • Streamlit • NumPy • Pandas • Plotly • Computer Vision • Facial Analysis

## Important limitation

Facial emotion classification is an inference from visual patterns and can be inaccurate because of lighting, camera quality, occlusion, pose, cultural/contextual differences and model limitations. Do not use the output as a definitive measure of a person's actual emotional state or for high-stakes decisions.
