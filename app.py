import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tempfile
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="🎬 Advanced Video Emotion Detection", layout="wide")

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/Users/koushal/Desktop/desktop/ml2 project/saved_model/best_emotion_cnn.keras"
    )

model = load_model()
class_names = ["Happy", "Neutral", "Sad"]

# ----------------- Helper Functions -----------------
def predict_emotion(frame):
    """Predict emotion from a BGR frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((128, 128))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds))
    return class_names[idx], float(np.max(preds)), preds[0]

def smooth_predictions(pred_list, window=3):
    """Smooth predictions over time using moving average of probabilities."""
    smoothed = []
    for i in range(len(pred_list)):
        start = max(0, i - window + 1)
        avg_probs = np.mean(pred_list[start:i+1], axis=0)
        smoothed.append((class_names[np.argmax(avg_probs)], float(np.max(avg_probs))))
    return smoothed

# ----------------- Streamlit UI -----------------
st.title("🎥 Advanced Video Emotion Detection")
st.write("Upload a video to analyze emotions frame-by-frame and visualize emotional trends.")

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("🚀 Start Emotion Analysis"):
        st.info("⏳ Processing your video... please wait.")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps) if fps > 0 else 0

        st.write(f"🎞️ **Video Duration:** {duration} seconds | **FPS:** {fps:.2f}")

        raw_preds = []
        frame_placeholder = st.image([])
        progress_bar = st.progress(0)

        # ------------- Frame-by-frame Analysis -------------
        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            label, conf, probs = predict_emotion(frame)
            raw_preds.append(probs)

            cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Second {sec + 1}: {label}")
            progress_bar.progress((sec + 1) / duration)

        cap.release()

        # ------------- Apply Smoothing -------------
        smoothed = smooth_predictions(raw_preds, window=3)
        final_preds = [(i+1, emo, conf) for i, (emo, conf) in enumerate(smoothed)]

        # ------------- Data Summary -------------
        df = pd.DataFrame(final_preds, columns=["Second", "Emotion", "Confidence"])
        st.subheader("🧾 Detailed Emotion Timeline")
        st.dataframe(df, use_container_width=True)

        # ------------- Emotion Statistics -------------
        emotion_counts = Counter(df["Emotion"])
        dominant, freq = emotion_counts.most_common(1)[0]
        st.success(f"✅ **Dominant Emotion:** `{dominant}` ({freq}/{len(df)} seconds)")

        # ------------- Charts -------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Emotion Frequency Pie Chart")
            fig1, ax1 = plt.subplots()
            ax1.pie(emotion_counts.values(), labels=emotion_counts.keys(),
                    autopct='%1.1f%%', startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

        with col2:
            st.subheader("📈 Confidence Trend")
            fig2, ax2 = plt.subplots()
            ax2.plot(df["Second"], df["Confidence"], color='blue', marker='o')
            ax2.set_xlabel("Seconds")
            ax2.set_ylabel("Confidence Level")
            ax2.set_ylim([0, 1])
            ax2.grid(True)
            st.pyplot(fig2)

        st.subheader("📊 Emotion Frequency Bar Chart")
        fig3, ax3 = plt.subplots()
        ax3.bar(emotion_counts.keys(), emotion_counts.values(), color=['#90CAF9', '#A5D6A7', '#FFAB91'])
        ax3.set_xlabel("Emotion")
        ax3.set_ylabel("Count")
        ax3.set_title("Emotion Frequency")
        st.pyplot(fig3)

        # ------------- CSV Export -------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Emotion Report (CSV)",
            data=csv,
            file_name="emotion_timeline.csv",
            mime="text/csv"
        )

        # ------------- Cleanup -------------
        try:
            os.remove(video_path)
        except Exception:
            pass

else:
    st.info("📁 Please upload a video file to begin.")