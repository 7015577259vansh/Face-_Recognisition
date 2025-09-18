#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os

# --------- KNN ALGO ----------
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


# --------- LOAD TRAINING DATA ----------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

dataset_path = "face_data"
facedata = []
labels = []
class_id = 0
names = {}  # id -> name mapping

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print("Found file:", fx)
        names[class_id] = fx[:-4]  # remove .npy
        data_item = np.load(os.path.join(dataset_path, fx))
        facedata.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

face_dataset = np.concatenate(facedata, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("Training data loaded.")
print("Trainset shape:", trainset.shape)


# --------- REAL TIME TESTING ----------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        predicted_name = names[int(out)]

        # Show name + rectangle
        cv2.putText(frame, predicted_name, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




