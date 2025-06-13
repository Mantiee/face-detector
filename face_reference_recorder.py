import cv2
import os
import time
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis
import configparser

# === Wczytanie konfiguracji ===
CONFIG_FILE = "recorder_config.cfg"
config = configparser.ConfigParser()

if not os.path.exists(CONFIG_FILE):
    print(f"[ERROR] Config file '{CONFIG_FILE}' not found.")
    exit()

config.read(CONFIG_FILE)

try:
    section = config["SETTINGS"]
    CAMERA_INDEX = int(section.get("CAMERA_INDEX", "0"))
    RECORD_SECONDS = int(section.get("RECORD_SECONDS", "30"))
    SAVE_FACES = section.get("SAVE_FACES", "true").lower() == "true"
except (KeyError, ValueError) as e:
    print(f"[ERROR] Invalid config value: {e}")
    exit()

# Foldery i pliki
OUTPUT_DIR = "reference_photos"
PICKLE_FILE = "reference_embeddings.pkl"
if SAVE_FACES:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicjalizacja InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Inicjalizacja kamery
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Can't use the camera.")
    exit()

print(f"[INFO] Recording for {RECORD_SECONDS} seconds...")
start_time = time.time()
embeddings = []

while time.time() - start_time < RECORD_SECONDS:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No frame.")
        break

    faces = app.get(frame)
    print(f"[DEBUG] Detected {len(faces)} face(s).")

    for i, face in enumerate(faces):
        emb = face.embedding
        embeddings.append(emb)

        if SAVE_FACES:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            face_img = frame[y1:y2, x1:x2]
            filename = os.path.join(OUTPUT_DIR, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[V] Saved: {filename}")

    cv2.imshow("Recording reference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] User interrupt.")
        break

cap.release()
cv2.destroyAllWindows()

# Zapis do pliku pickle
if embeddings:
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Saved {len(embeddings)} face embeddings to {PICKLE_FILE}")
else:
    print("[ERROR] No faces found.")
