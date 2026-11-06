# Face Recognition (OpenCV + KNN) - macOS

## Setup (once)
```bash
cd face_recognition_mac
./setup_mac.sh
source .venv/bin/activate
```

## Use
```bash
python3 face_app.py collect            # asks for a name, press q when done
python3 face_app.py collect --name Vansh
python3 face_app.py recognize          # press q to quit
python3 face_app.py recognize --threshold 6000   # label far-away matches "Unknown"
python3 face_app.py list
python3 face_app.py --camera 1 recognize         # use another camera
```
Collect 2+ people (or 1 person under different lighting) for best results.

## macOS camera permission
First run, macOS asks for camera access for Terminal (or VS Code/PyCharm).
If it doesn't: System Settings > Privacy & Security > Camera > enable it, then fully restart the app.
If the iPhone camera opens instead of the Mac one, use `--camera 1`.
