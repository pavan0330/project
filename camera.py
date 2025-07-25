import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from playsound import playsound
import threading
import time
import os

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.yolo = YOLO("yolov8n.pt")

        self.last_alert_time = 0
        self.alert_interval = 10  # seconds between alerts
        self.lock = threading.Lock()  # Prevent concurrent access to camera

    def __del__(self):
        self.video.release()

    def send_alert(self, message):
        if time.time() - self.last_alert_time < self.alert_interval:
            return
        self.last_alert_time = time.time()

        def alert_thread():
            print(f"🚨 ALERT: {message}")
            try:
                playsound('alert.wav')  # Use absolute path if needed
            except Exception as e:
                print(f"Sound error: {e}")

            try:
                sender_email = "pavangujjarlapudi1@gmail.com"
                receiver_email = "pavangujjarlapudi1@gmail.com"
                password = os.getenv("EMAIL_APP_PASSWORD")  # Use environment variable

                if not password:
                    print("⚠️ Email password not set in environment variable.")
                    return

                msg = MIMEText(message)
                msg["Subject"] = "CCTV Alert"
                msg["From"] = sender_email
                msg["To"] = receiver_email

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, msg.as_string())
                print("✅ Email sent.")
            except Exception as e:
                print(f"❌ Email failed: {e}")

        threading.Thread(target=alert_thread).start()

    def detect_behavior(self, landmarks):
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        except IndexError:
            return "Invalid landmarks"

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2

        if shoulder_y > hip_y:
            self.send_alert("Fall Detected!")
            return "Fall Detected"

        if left_hand.y < left_shoulder.y and right_hand.y < right_shoulder.y:
            self.send_alert("Hands Raised (Potential Threat)")
            return "Hands Raised"

        return "Normal"

    def detect_weapons(self, frame):
        results = self.yolo(frame, verbose=False)[0]
        detected_labels = []

        for box in results.boxes:
            if box.cls is None or len(box.cls) == 0:
                continue

            cls_id = int(box.cls[0])
            label = results.names[cls_id]

            if label.lower() in ["knife", "scissors", "gun"]:
                detected_labels.append(label)
                self.send_alert(f"Weapon Detected: {label}")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return detected_labels

    def get_frame(self):
        with self.lock:
            success, frame = self.video.read()

        if not success:
            return None

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)

        status = "No person detected"
        if result.pose_landmarks:
            try:
                landmarks = result.pose_landmarks.landmark
                status = self.detect_behavior(landmarks)
                self.mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            except Exception as e:
                print(f"Pose error: {e}")
                status = "Pose Error"

        weapons = self.detect_weapons(frame)
        if weapons:
            status += " | Weapon: " + ", ".join(weapons)

        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
