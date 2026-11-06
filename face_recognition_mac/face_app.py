#!/usr/bin/env python3
"""
Face Recognition (OpenCV + KNN) - macOS-ready version.

Usage:
    python3 face_app.py collect      # capture faces for a person
    python3 face_app.py recognize    # live recognition
    python3 face_app.py list         # show saved people

Press 'q' in the video window to quit.
Dataset format is the same as the original repo (face_data/<name>.npy),
so your existing data keeps working.
"""
import argparse
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "face_data")
FACE_SIZE = (100, 100)
OFFSET = 10
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)


# ---------------- camera ----------------
def open_camera(index=0):
    # AVFoundation is the native macOS backend
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" \
        else cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            "Could not open camera.\n"
            "- macOS: System Settings > Privacy & Security > Camera > enable your Terminal/IDE, then restart it.\n"
            "- Try a different index: --camera 1 (index 0 may be an iPhone via Continuity Camera).\n"
            "- Make sure no other app is using the camera."
        )
    return cap


def read_frame(cap, retries=30):
    for _ in range(retries):
        ret, frame = cap.read()
        if ret:
            return frame
    sys.exit("Camera opened but returned no frames (check camera permission).")


# ---------------- face helpers ----------------
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.3, 5)
    return sorted(faces, key=lambda f: f[2] * f[3])  # smallest -> largest


def crop_face(frame, x, y, w, h):
    """Crop with padding, clamped to the frame (original crashed near edges)."""
    H, W = frame.shape[:2]
    x1, y1 = max(x - OFFSET, 0), max(y - OFFSET, 0)
    x2, y2 = min(x + w + OFFSET, W), min(y + h + OFFSET, H)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, FACE_SIZE)


# ---------------- KNN (vectorised) ----------------
def knn(train, test, k=5):
    """train: N x (D+1) with label in last column. Returns (label, mean_distance)."""
    X, y = train[:, :-1], train[:, -1]
    d = np.linalg.norm(X - test, axis=1)
    k = min(k, len(d))
    idx = np.argsort(d)[:k]
    labels, counts = np.unique(y[idx], return_counts=True)
    best = labels[np.argmax(counts)]
    return best, float(d[idx][y[idx] == best].mean())


def load_dataset():
    if not os.path.isdir(DATASET_PATH):
        sys.exit(f"No dataset folder at {DATASET_PATH}. Run 'collect' first.")
    faces, labels, names = [], [], {}
    for cid, fname in enumerate(sorted(f for f in os.listdir(DATASET_PATH) if f.endswith(".npy"))):
        data = np.load(os.path.join(DATASET_PATH, fname)).astype(np.float32)
        names[cid] = fname[:-4]
        faces.append(data)
        labels.append(np.full((data.shape[0], 1), cid, dtype=np.float32))
        print(f"Loaded {fname}: {data.shape[0]} samples")
    if not faces:
        sys.exit("No .npy files in face_data/. Run 'collect' first.")
    return np.concatenate([np.concatenate(faces), np.concatenate(labels)], axis=1), names


# ---------------- modes ----------------
def collect(args):
    os.makedirs(DATASET_PATH, exist_ok=True)
    name = args.name or input("Enter the name of the person: ").strip()
    if not name:
        sys.exit("Name cannot be empty.")
    cap = open_camera(args.camera)
    samples, skip = [], 0
    print(f"Collecting for '{name}'. Look at the camera, move your head a little. Press 'q' to finish.")
    while True:
        frame = read_frame(cap)
        faces = detect_faces(frame)
        if faces:
            x, y, w, h = faces[-1]  # largest face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = crop_face(frame, x, y, w, h)
            skip += 1
            if face is not None and skip % 10 == 0:
                samples.append(face)
                print(f"Collected samples: {len(samples)}")
            if face is not None:
                cv2.imshow("Face Section", face)
        cv2.putText(frame, f"Samples: {len(samples)} (q to finish)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not samples:
        sys.exit("No face samples captured - nothing saved.")
    arr = np.asarray(samples).reshape(len(samples), -1)
    out = os.path.join(DATASET_PATH, name + ".npy")
    np.save(out, arr)
    print(f"Saved {arr.shape[0]} samples -> {out}")


def recognize(args):
    trainset, names = load_dataset()
    print("Trainset shape:", trainset.shape)
    cap = open_camera(args.camera)
    while True:
        frame = read_frame(cap)
        for (x, y, w, h) in detect_faces(frame):
            face = crop_face(frame, x, y, w, h)
            if face is None:
                continue
            label, dist = knn(trainset, face.flatten().astype(np.float32), k=args.k)
            text = names[int(label)]
            if args.threshold and dist > args.threshold:
                text = "Unknown"
            colour = (0, 255, 0) if text != "Unknown" else (0, 0, 255)
            cv2.putText(frame, text, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.imshow("Face Recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def list_people(_):
    if not os.path.isdir(DATASET_PATH):
        print("No people saved yet.")
        return
    for f in sorted(os.listdir(DATASET_PATH)):
        if f.endswith(".npy"):
            print(f"- {f[:-4]}: {np.load(os.path.join(DATASET_PATH, f)).shape[0]} samples")


def main():
    p = argparse.ArgumentParser(description="Face recognition (OpenCV + KNN)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="capture face samples")
    c.add_argument("--name", help="person's name")
    c.set_defaults(fn=collect)
    r = sub.add_parser("recognize", help="live recognition")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--threshold", type=float, default=0,
                   help="distance above which a face is 'Unknown' (0 = off)")
    r.set_defaults(fn=recognize)
    l = sub.add_parser("list", help="list saved people")
    l.set_defaults(fn=list_people)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
