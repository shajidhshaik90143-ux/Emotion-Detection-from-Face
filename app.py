import os
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px

from emotion_engine import analyze_image, draw_results

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
HISTORY_FILE = DATA_DIR / "emotion_history.csv"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="EmotionVision AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 17px; opacity: .75; margin-top: 0;}
.metric-card {padding: 18px; border-radius: 16px; border: 1px solid rgba(128,128,128,.25);}
.small-note {font-size: 13px; opacity: .7;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">😊 EmotionVision AI</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time facial emotion analysis with face detection, confidence scores, analytics and history.</p>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading emotion AI models...")
def load_engine():
    # Importing DeepFace here keeps initial page loading cleaner.
    from deepface import DeepFace
    return DeepFace

try:
    DeepFace = load_engine()
except Exception as e:
    st.error("AI model initialization failed.")
    st.code(str(e))
    st.info("Use Python 3.10/3.11, then run: pip install -r requirements.txt")
    st.stop()

def save_history(results, source):
    if not results:
        return
    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        rows.append({
            "timestamp": now,
            "source": source,
            "emotion": r["emotion"],
            "confidence": round(float(r["confidence"]), 2),
            "face_index": r["face_index"],
        })
    exists = HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=["timestamp","source","emotion","confidence","face_index"])
    return pd.read_csv(HISTORY_FILE)

with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Detection mode", ["Upload Image", "Camera"])
    detector_backend = st.selectbox(
        "Face detector",
        ["opencv", "retinaface", "mtcnn"],
        index=0,
        help="OpenCV is fastest. RetinaFace/MT-CNN can be more accurate but slower."
    )
    min_conf = st.slider("Minimum confidence (%)", 0, 100, 0, 5)
    show_box = st.checkbox("Show face boxes", True)
    show_label = st.checkbox("Show emotion labels", True)
    save_results = st.checkbox("Save results to history", True)
    st.divider()
    st.caption("Model: DeepFace emotion analysis")
    st.caption("For demos/research; not a diagnostic system.")

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png","webp"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing faces and emotions..."):
            results = analyze_image(frame, DeepFace, detector_backend)

        results = [r for r in results if r["confidence"] >= min_conf]
        annotated = draw_results(frame.copy(), results, show_box, show_label)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        left, right = st.columns([1.5, 1])
        with left:
            st.image(annotated_rgb, caption="Emotion analysis result", use_container_width=True)
        with right:
            st.subheader("Detection Summary")
            st.metric("Faces detected", len(results))
            if results:
                counts = pd.Series([r["emotion"] for r in results]).value_counts()
                dominant = counts.index[0]
                st.metric("Dominant emotion", dominant.title())
                st.metric("Average confidence", f"{np.mean([r['confidence'] for r in results]):.1f}%")

                table = pd.DataFrame([{
                    "Face": r["face_index"],
                    "Emotion": r["emotion"].title(),
                    "Confidence": f"{r['confidence']:.1f}%"
                } for r in results])
                st.dataframe(table, hide_index=True, use_container_width=True)

                if save_results:
                    save_history(results, "image")
            else:
                st.warning("No face passed the selected confidence threshold.")

elif mode == "Camera":
    st.info("Allow camera access when prompted. Camera mode analyzes frames captured by Streamlit.")
    camera = st.camera_input("Take a photo for emotion analysis")
    if camera:
        pil_img = Image.open(camera).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        with st.spinner("Analyzing camera frame..."):
            results = analyze_image(frame, DeepFace, detector_backend)
        results = [r for r in results if r["confidence"] >= min_conf]
        annotated = draw_results(frame.copy(), results, show_box, show_label)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        if results:
            cols = st.columns(min(4, len(results)))
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    st.metric(f"Face {r['face_index']}", r["emotion"].title(), f"{r['confidence']:.1f}%")
            if save_results:
                save_history(results, "camera")
        else:
            st.warning("No face detected.")

st.divider()
st.header("📊 Emotion Analytics")

history = load_history()
if history.empty:
    st.info("No saved detections yet. Run an analysis with 'Save results to history' enabled.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", len(history))
    c2.metric("Unique emotions", history["emotion"].nunique())
    c3.metric("Avg confidence", f"{history['confidence'].mean():.1f}%")
    c4.metric("Most frequent", history["emotion"].value_counts().idxmax().title())

    a, b = st.columns(2)
    with a:
        counts = history["emotion"].value_counts().reset_index()
        counts.columns = ["emotion", "count"]
        fig = px.bar(counts, x="emotion", y="count", title="Emotion Frequency", text="count")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig2 = px.pie(history, names="emotion", title="Emotion Distribution", hole=.35)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent Detection History")
    st.dataframe(history.tail(100).iloc[::-1], hide_index=True, use_container_width=True)

    csv_data = history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History CSV", csv_data, "emotion_history.csv", "text/csv")

    if st.button("🗑️ Clear History"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.rerun()

st.divider()
st.caption("EmotionVision AI • Educational computer-vision project • Predictions can be imperfect and should not be used for high-stakes decisions.")
