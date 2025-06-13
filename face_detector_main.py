import cv2
import os
import pickle
import numpy as np
import time
import subprocess
from insightface.app import FaceAnalysis
import threading
import keyboard
import configparser

# Kolory ANSI
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

# === Wczytanie konfiguracji ===
config = configparser.ConfigParser()
if not os.path.exists("main_config.cfg"):
    print(f"{RED}[ERROR] File 'main_config.cfg' not found.{RESET}")
    exit()

config.read("main_config.cfg")
try:
    section = config["SETTINGS"]
    TOLERANCE = float(section.get("TOLERANCE", "0.4"))
    CAMERA_INDEX = int(section.get("CAMERA_INDEX", "0"))
    TIMEOUT_SECONDS = int(section.get("TIMEOUT_SECONDS", "5"))
    REQUIRED_CONSECUTIVE_OK = int(section.get("REQUIRED_CONSECUTIVE_OK", "3"))
    gui_mode = int(section.get("gui_mode", "1"))
    face_mode = int(section.get("face_mode", "1"))
    mode = int(section.get("mode", "1"))
except (ValueError, KeyError) as e:
    print(f"{RED}[ERROR] Invalid value in main_config.cfg: {e}{RESET}")
    exit()

INTERVAL = 1.0
DISPLAY_DURATION = 1.2
EMBEDDINGS_PATH = "reference_embeddings.pkl"
use_gui = (gui_mode == 1)

if not os.path.exists(EMBEDDINGS_PATH):
    print(f"[ERROR] File '{EMBEDDINGS_PATH}' not found.")
    exit()

with open(EMBEDDINGS_PATH, "rb") as f:
    reference_embeddings = pickle.load(f)

if not reference_embeddings:
    print("[ERROR] No embeddings.")
    exit()

reference_embeddings = np.array(reference_embeddings)
reference_embeddings = reference_embeddings / np.linalg.norm(reference_embeddings, axis=1, keepdims=True)

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Cant use the camera.")
    exit()

last_check = 0
last_my_face_time = time.time()
screen_black = False
all_faces_info = []
simulated_action_triggered = False
consecutive_ok = 0

def is_my_face(embedding):
    emb = embedding / np.linalg.norm(embedding)
    sims = np.dot(reference_embeddings, emb)
    best_sim = np.max(sims)
    return (1 - best_sim) < TOLERANCE, 1 - best_sim

def turn_off_monitor():
    ps = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class MonitorControl {
    [DllImport("user32.dll")]
    public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);
}
"@
[MonitorControl]::SendMessage(-1, 0x0112, 0xF170, 2)
"""
    subprocess.Popen(["powershell", "-Command", ps], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wake_up_monitor():
    ps = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class PowerControl {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_CONTINUOUS = 0x80000000;
}
"@
[PowerControl]::SetThreadExecutionState([PowerControl]::ES_DISPLAY_REQUIRED -bor [PowerControl]::ES_SYSTEM_REQUIRED)
Start-Sleep -Milliseconds 100
[PowerControl]::SetThreadExecutionState([PowerControl]::ES_CONTINUOUS)
"""
    subprocess.Popen(["powershell", "-Command", ps], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def unmount_veracrypt():
    try:
        subprocess.run(["veracrypt", "/dismount", "/force", "/quit"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{BLUE}[INFO] VeraCrypt unmounted.{RESET}")
    except FileNotFoundError:
        print("[ERROR] VeraCrypt not found in PATH.")

def start_failsafe_listener():
    def listen():
        keyboard.wait("ctrl+m")
        print(f"{BLUE}[FAILSAFE] Clicked CTRL+M – killing the program.{RESET}")
        os._exit(0)

    threading.Thread(target=listen, daemon=True).start()

start_failsafe_listener()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No camera output.")
        break

    current_time = time.time()

    if current_time - last_check >= INTERVAL:
        last_check = current_time
        faces = app.get(frame)
        print(f"[DEBUG] Detected {len(faces)} Face(s).")

        my_face_detected = False
        max_area = 0
        max_area_match = False
        all_faces_info = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            area = (x2 - x1) * (y2 - y1)
            match, acc = is_my_face(face.embedding)
            label = f"{'DETECTED' if match else 'NOT'} ({acc:.3f})"
            color = GREEN if match else RED
            print(f"{color}[DEBUG] Face: accuracy={acc:.3f} | distance={1-acc:.3f} | area={area} → {'V' if match else 'X'}{RESET}")
            color_bgr = (0, 255, 0) if match else (0, 0, 255)
            all_faces_info.append(((x1, y1, x2, y2), label, color_bgr))

            if face_mode == 1 and match:
                my_face_detected = True
            if face_mode == 2 and area > max_area:
                max_area = area
                max_area_match = match

        if face_mode == 2:
            my_face_detected = max_area_match

        if my_face_detected:
            last_my_face_time = current_time
            consecutive_ok += 1
            if consecutive_ok >= REQUIRED_CONSECUTIVE_OK:
                if mode == 1 and simulated_action_triggered:
                    print(f"{BLUE}[INFO] Would return form an action...{RESET}")
                    simulated_action_triggered = False
                if screen_black and mode == 2:
                    print(f"{BLUE}[INFO] Detected face {REQUIRED_CONSECUTIVE_OK} in a row – turning on monitor.{RESET}")
                    wake_up_monitor()
                    screen_black = False
        else:
            consecutive_ok = 0

    if time.time() - last_my_face_time > TIMEOUT_SECONDS:
        if mode == 1 and not simulated_action_triggered:
            print(f"{BLUE}[INFO] Would do an action...{RESET}")
            simulated_action_triggered = True
        elif mode == 2 and not screen_black:
            print(f"{BLUE}[INFO] Didn't detect your face – turning off monitor.{RESET}")
            screen_black = True
            turn_off_monitor()
        elif mode == 3:
            print(f"{BLUE}[INFO] Timeout – Unmounting VeraCrypt and hibernating.{RESET}")
            unmount_veracrypt()
            subprocess.run("shutdown /h", shell=True)
            exit()

    if screen_black and mode == 2:
        key = cv2.waitKey(1)
        if key != -1:
            print(f"{BLUE}[INFO] Turning on monitor by user interrupt.{RESET}")
            break
        continue

    if use_gui:
        for bbox, label, color in all_faces_info:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

        mode_label = f"Mode: {['', 'Debug', 'Monitor', 'VeraCrypt'][mode]}"
        cv2.putText(frame, mode_label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
