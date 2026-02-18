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
import torchvision.models as models
import torch.nn as nn

st.set_page_config(page_title="Deepfake Detection", layout="wide")

ADMIN_EMAIL = "admin@deepfake.com"
ADMIN_PASSWORD = "admin123"

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
    sender_email = "deepfakealert@gmail.com"
    app_password = "hzxu tzdo unfx edqu"

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
    except Exception as e:
        st.error("Email sending failed")




# ---------- LOGIN FUNCTION ----------
def login_page():
    st.title("Login")

    email = st.text_input("Enter your Email")
    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):

        if email == "":
            st.warning("Please enter your email")
            return

        # -------- ADMIN LOGIN --------
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            st.session_state["logged_in"] = True
            st.session_state["is_admin"] = True
            st.success("Admin Login Successful")
            st.rerun()

        # -------- USER LOGIN --------
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

    if users:
        for user in users:
            st.write(f" {user[0]} |  {user[1]}")
    else:
        st.info("No users registered yet.")
# If admin, don't show detection system
if st.session_state["is_admin"]:
    st.stop()

FRAMES_DIR = "frames"
FACE_DIR = "face_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

def clear_old_frames():
    for folder in [FRAMES_DIR, FACE_DIR]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ------------------ CNN MODEL LOAD ------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)

    # Replace last layer for binary classification (REAL / FAKE)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model.eval()
    return model

model = load_model()

def fake_prediction(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "FAKE" if predicted.item() == 1 else "REAL"
    return label, confidence.item()

# ------------------ FUNCTIONS ------------------
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


def detect_faces():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved = 0
    for img_name in os.listdir(FRAMES_DIR):
        img_path = os.path.join(FRAMES_DIR, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_path = os.path.join(FACE_DIR, f"face_{saved}.jpg")
            cv2.imwrite(face_path, face)
            saved += 1

    return saved

# ------------------ UI ------------------
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

        # ---- SHOW FACE FRAMES ----
        st.subheader(" Extracted Face Frames")
        cols = st.columns(4)

        images = os.listdir(FACE_DIR)
        for i, img in enumerate(images[:12]):
            img_path = os.path.join(FACE_DIR, img)
            cols[i % 4].image(img_path, caption=img, use_container_width=True)

        # ---- PREDICTION ----
        st.subheader(" Final Prediction")

        results = []
        confidences = []

        for img in images:
            img_path = os.path.join(FACE_DIR, img)
            label, conf = fake_prediction(img_path)
            results.append(label)
            confidences.append(conf)

       
        # ---- FINAL CALCULATION ----
        if len(confidences) > 0:
            fake_count = results.count("FAKE")
            real_count = results.count("REAL")
            final_result = "FAKE" if fake_count > real_count else "REAL"
            avg_conf = sum(confidences) / len(confidences)

            # -------- EMAIL ALERT (FAKE ONLY) --------
            if final_result == "FAKE" and "user_email" in st.session_state:
                send_email_alert(
                    to_email=st.session_state["user_email"],
                    result=final_result,
                    confidence=avg_conf
                )

            st.markdown(f"""
            ###  Final Video Result
            - **Prediction:** `{final_result}`
            - **Average Confidence:**
            {round(avg_conf*100, 2)} %
            """)
