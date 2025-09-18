#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

skip = 0
facedata = []
dataset_path = 'face_data/'  # Make sure this folder exists

file_name = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])  # Sort by area
    

    # Pick the largest face
    for (x, y, w, h) in faces[-1:]:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract face (ROI)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            facedata.append(face_section)
            print(f"Collected samples: {len(facedata)}")

    # Show frames
    cv2.imshow("Frame", frame)
    if len(facedata) > 0:
        cv2.imshow("Face Section", face_section)

    # Exit
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert list to numpy array and reshape
facedata = np.asarray(facedata)
facedata = facedata.reshape((facedata.shape[0], -1))
print(facedata.shape)

# Save the data
np.save(dataset_path + file_name + '.npy', facedata)
print("Data successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()


# In[ ]:




