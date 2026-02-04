import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
IMAGE_PATH = "What-is-Facial-Recognition.webp"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
model = load_model("face_recognition_mobilenetv2.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding:30px">
        <h1 style="font-size:48px">📸 Face Attendance System</h1>
        <p style="font-size:20px; color:gray">
            AI-powered attendance using MobileNetV2
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(IMAGE_PATH, width="stretch")

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.radio(
    "Choose Section",
    ["🏫 Mark Attendance", "📥 Download Attendance"]
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "attendance" not in st.session_state:
    st.session_state.attendance = []

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    for row in st.session_state.attendance:
        if row["Name"] == name and row["Date"] == date:
            return

    st.session_state.attendance.append({
        "Name": name,
        "Date": date,
        "Time": time
    })

# -----------------------------
# MARK ATTENDANCE (CLOUD SAFE)
# -----------------------------
if menu == "🏫 Mark Attendance":

    st.info("📷 Capture an image using your camera")

    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)
            idx = np.argmax(preds)
            confidence = preds[0][idx]

            if confidence > CONF_THRESHOLD:
                name = labels[idx]
                color = GREEN
                mark_attendance(name)
                label = f"{name}  {confidence:.2f}"
            else:
                name = "UNKNOWN"
                color = RED
                label = f"UNKNOWN  {confidence:.2f}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x, y - th - 15),
                (x + tw + 10, y),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2,
                cv2.LINE_AA
            )

        st.image(frame, channels="BGR", caption="Processed Frame", width="stretch")

# -----------------------------
# DOWNLOAD ATTENDANCE
# -----------------------------
elif menu == "📥 Download Attendance":

    if len(st.session_state.attendance) == 0:
        st.warning("No attendance recorded.")
    else:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df, width="stretch")

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "attendance.csv"
        )

st.divider()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px">
        <b>Face Recognition Attendance System</b><br>
        Built with Streamlit • OpenCV • MobileNetV2
    </div>
    """,
    unsafe_allow_html=True
)
