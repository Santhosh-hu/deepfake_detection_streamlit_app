import sqlite3
import smtplib
from email.message import EmailMessage
import streamlit as st
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import re
from mtcnn import MTCNN

st.set_page_config(page_title="Deepfake Detection", layout="wide")

ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]

# ---------- DATABASE SETUP ----------
def get_db_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_users_table()

# ================= EMAIL ALERT FUNCTION =================
def send_email_alert(to_email, result, confidence):

    sender_email = st.secrets["EMAIL"]
    app_password = st.secrets["APP_PASSWORD"]

    msg = EmailMessage()
    msg["Subject"] = "Deepfake Detection Alert"
    msg["From"] = sender_email
    msg["To"] = to_email

    msg.set_content(f"""
Hello,

Your video has been analyzed.

Result       : {result}
Confidence   : {round(confidence * 100, 2)} %

Thank you for using Deepfake Detection System.
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
    except:
        st.error("Email sending failed")

# ---------- LOGIN FUNCTION ----------
def login_page():

    st.title("Login")

    email = st.text_input("Enter your Email")

    col1, col2 = st.columns(2)

    with col1:
        user_login = st.button("User Login")

    with col2:
        admin_login = st.button("Login as Admin")

    if user_login:

        if email == "":
            st.warning("Please enter your email")
            return

        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_pattern, email):
            st.error("Invalid Email Format")
            return

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (email) VALUES (?)",
                (email,)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()

        st.session_state["logged_in"] = True
        st.session_state["is_admin"] = False
        st.session_state["user_email"] = email
        st.success("User Login Successful")
        st.rerun()

    if admin_login:

        admin_password = st.text_input(
            "Enter Admin Password",
            type="password",
            key="admin_pass"
        )

        if admin_password:

            if admin_password == ADMIN_PASSWORD:
                st.session_state["logged_in"] = True
                st.session_state["is_admin"] = True
                st.success("Admin Login Successful")
                st.rerun()
            else:
                st.error("Invalid Admin Password")

# ---------- SESSION CHECK ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

st.title(" Deepfake Detection Dashboard")

if st.session_state["is_admin"]:
    st.sidebar.success("Logged in as Admin")
else:
    st.sidebar.success("Logged in as User")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["is_admin"] = False
    st.rerun()

# ---------- ADMIN VIEW USERS ----------
if st.session_state["is_admin"]:

    st.subheader("Registered Users")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, created_at FROM users")
    users = cursor.fetchall()
    conn.close()

    for user in users:
        st.write(f"{user[0]} | {user[1]}")

    st.stop()

# ---------- FOLDERS ----------
FRAMES_DIR = "frames"
FACE_DIR = "face_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

def clear_old_frames():
    for folder in [FRAMES_DIR, FACE_DIR]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    model.eval()
    return model

model = load_model()

# ---------- MTCNN ----------
detector = MTCNN()

def fake_prediction(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "FAKE" if predicted.item() == 1 else "REAL"
    return label, confidence.item()

# ---------- FRAME EXTRACTION ----------
def extract_frames(video_path, max_frames=30):

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened() and count < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(FRAMES_DIR, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return count

# ---------- FACE DETECTION ----------
def detect_faces():

    saved = 0

    for img_name in os.listdir(FRAMES_DIR):

        img_path = os.path.join(FRAMES_DIR, img_name)
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb)

        for face in faces:

            x,y,w,h = face["box"]
            crop = img[y:y+h, x:x+w]

            if crop.size > 0:

                face_path = os.path.join(FACE_DIR, f"face_{saved}.jpg")
                cv2.imwrite(face_path, crop)
                saved += 1

    return saved

# ---------- UI ----------
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    st.success("Video uploaded successfully")

    if st.button(" Start Deepfake Detection"):

        clear_old_frames()

        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.info(" Extracting frames...")
        frame_count = extract_frames("temp.mp4")

        st.info(" Detecting faces...")
        face_count = detect_faces()

        st.success(f"Frames: {frame_count} | Faces detected: {face_count}")

        st.subheader(" Extracted Face Frames")

        cols = st.columns(4)
        images = os.listdir(FACE_DIR)

        for i, img in enumerate(images[:12]):

            img_path = os.path.join(FACE_DIR, img)
            cols[i % 4].image(img_path, caption=img, use_container_width=True)

        st.subheader(" Final Prediction")

        results = []
        confidences = []

        for img in images:

            img_path = os.path.join(FACE_DIR, img)
            label, conf = fake_prediction(img_path)
            results.append(label)
            confidences.append(conf)

        if len(confidences) > 0:

            fake_count = results.count("FAKE")
            real_count = results.count("REAL")

            final_result = "FAKE" if fake_count > real_count else "REAL"
            avg_conf = sum(confidences) / len(confidences)

            if final_result == "FAKE" and "user_email" in st.session_state:

                send_email_alert(
                    to_email=st.session_state["user_email"],
                    result=final_result,
                    confidence=avg_conf
                )

            st.markdown(f"""
###  Final Video Result
- **Prediction:** `{final_result}`
- **Average Confidence:** {round(avg_conf*100,2)} %
""")
