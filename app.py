import os
import cv2
import pickle
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime, timedelta
import streamlit as st
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
from mtcnn import MTCNN
import time

THRESHOLD = 1

SAVE_IMAGE_FOLDER = "attendance_images"
ID_CARD_FOLDER = "idcards"

if not os.path.exists(SAVE_IMAGE_FOLDER):
    os.makedirs(SAVE_IMAGE_FOLDER)
if not os.path.exists(ID_CARD_FOLDER):
    os.makedirs(ID_CARD_FOLDER)

# Load FaceNet model and MTCNN detector
MyFaceNet = FaceNet()
detector = MTCNN()

# Load face database from model.pkl
with open("model.pkl", "rb") as f:
    database = pickle.load(f)

# SQLite database setup
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create attendance table if not exists
cursor.execute(""" 
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    nim TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL
)
""")
conn.commit()

# Function to mark attendance in SQLite
def mark_attendance(name, nim, frame):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")  # Format waktu yang digunakan saat ini

    # Get the last attendance record for the same NIM on the same day
    cursor.execute(
        "SELECT time FROM attendance WHERE nim = ? AND date = ? ORDER BY id DESC LIMIT 1", (nim, current_date)
    )
    result = cursor.fetchone()

    if result:
        last_attendance_time = result[0]
        # Format last_attendance_time jika formatnya HH-MM-SS
        try:
            if len(last_attendance_time.split('-')) == 3:  # Check if format is HH-MM-SS
                last_attendance_time = f"{current_date} {last_attendance_time}"  # Combine current date and time
            else:
                last_attendance_time = f"{current_date} {last_attendance_time}"

            # Coba konversi waktu dengan format yang benar
            last_time = datetime.strptime(last_attendance_time, "%Y-%m-%d %H-%M-%S")  # Perbaiki format
        except ValueError:
            # Jika formatnya tidak sesuai, log kesalahan
            st.warning(f"Waktu yang disimpan tidak sesuai format: {last_attendance_time}")
            return

        # Check if less than 1 minute has passed
        time_diff = now - last_time
        if time_diff < timedelta(minutes=60):
            st.info(f"{name} (NIM: {nim}) sudah tercatat beberapa saat yang lalu.")
            return

    # Record attendance
    cursor.execute(
        "INSERT INTO attendance (name, nim, date, time) VALUES (?, ?, ?, ?)",
        (name, nim, current_date, current_time),
    )
    conn.commit()

    # Displaying notification at the top of the frame
    notification_placeholder.success(f"Kehadiran tercatat: {name} (NIM: {nim}) pada {current_date} {current_time}")

    # Save image of the attendee with a unique filename
    image_name = f"{name}_{nim}_{current_time}.jpg"  # Nama file mengikuti waktu dari database
    image_path_to_save = os.path.join(SAVE_IMAGE_FOLDER, image_name)
    cv2.imwrite(image_path_to_save, frame)  # Save the current frame as an image
    st.success(f"Foto {name} (NIM: {nim}) berhasil disimpan di {image_path_to_save}")

    # Display ID card after a short delay (within 2 seconds)
    id_card_image_path = os.path.join(ID_CARD_FOLDER, f"{nim}_idcard.jpg")  # Assume ID card image is named by NIM
    if os.path.exists(id_card_image_path):
        # Show ID card in a small pop-up for 2 seconds
        id_card_image = Image.open(id_card_image_path)

        # Layout for showing ID card on the right
        col1, col2 = st.columns([2, 1])  # Adjust layout: ID card in the right column
        with col1:
            with st.empty():  # Use st.empty for pop-up behavior
                st.image(frame, caption=f"Camera Feed", use_column_width=True)
        with col2:
            st.image(id_card_image, caption=f"ID Card {name} (NIM: {nim})", width=150)
            st.info("Informasi Pengguna")  # Information message
            time.sleep(2)  # Delay for 2 seconds before hiding ID card
    else:
        st.warning(f"ID card untuk {name} (NIM: {nim}) tidak ditemukan.")

# Streamlit UI setup
st.set_page_config(page_title="Automated Attendance System", page_icon="🚀")

st.title("Automated Attendance System")
st.write("Gunakan aplikasi ini untuk mencatat kehadiran dengan pengenalan wajah.")

# Layout: Two buttons in one row
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("Start Kamera", key="start_camera_button")
with col2:
    stop_camera = st.button("Stop Kamera", key="stop_camera_button")

# Placeholder for notifications and video stream
notification_placeholder = st.empty()  # Notification appears here
stframe = st.empty()  # Placeholder for video frame

# Initialize session state for camera control
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Camera stream logic
if start_camera:
    st.session_state.camera_active = True
    cap = cv2.VideoCapture(0)  # Start video capture

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            notification_placeholder.error("Kamera tidak tersedia!")
            break

        # Face detection logic
        faces = detector.detect_faces(frame)

        for face in faces:
            x1, y1, width, height = face["box"]
            x2, y2 = x1 + width, y1 + height

            # Crop and preprocess the detected face
            cropped_face = frame[y1:y2, x1:x2]
            cropped_face = Image.fromarray(cropped_face).resize((160, 160))
            cropped_face = asarray(cropped_face)
            cropped_face = expand_dims(cropped_face, axis=0)

            # Generate embedding for the detected face
            signature = MyFaceNet.embeddings(cropped_face)

            # Identify the face
            min_dist = float("inf")
            identity = "Unknown"
            nim = "Unknown"

            for key, value in database.items():
                dist = np.linalg.norm(value["embedding"] - signature)
                if dist < min_dist:
                    min_dist = dist
                    if dist < THRESHOLD:
                        identity = value["name"]
                        nim = value["nim"]

            # Mark attendance if identity is recognized
            if identity != "Unknown":
                mark_attendance(identity, nim, frame)

            # Draw bounding box and identity
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(
                frame,
                f"{identity} ({nim})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

        # Convert frame from BGR to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    # Release resources when stopping
    cap.release()
    cv2.destroyAllWindows()

if stop_camera:
    st.session_state.camera_active = False
    notification_placeholder.info("Kamera dihentikan.")

# Close SQLite connection
conn.close()