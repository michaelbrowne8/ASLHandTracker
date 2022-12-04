import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from handDetector import handDetector as hd
import cv2
import time
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score
'''

Ok so I overcomplicated it, one layer works

File's a little messy, so here's whats going on.

    - dataNew.npy is unprocessed, so I process it all using preprocess
    - for some of the labels, there just isn't enough info on them
        (like c or o) so those just become "nothing"
        * scroll down to line 60 or so for new alphabet
    - train test split up the info, saving 20% for testing
    - creating the model (feel free to change around to see if it works better)
    - giving the model its parameters (batch size, lr, ...)
    - training and testing

At the very bottom of the file is the hand detector using the webcam
This one we can actually make into a separate file so that it can just be 
    imported and ran
But it should show the tracked hand and the letter it thinks it is

Press 'Q' to quit

'''

labels = np.load('labelsNew.npy') # 19436 labels
data = np.load('dataNew.npy').astype(float)    # 19436 preprocessed images 
                               # of x,y,z at 21 landmarks a hand

def preprocess(lmlist):
  max = lmlist.max(axis=0)
  min = lmlist.min(axis=0)
  newList = np.zeros_like(lmlist)

  if max[0] - min[0] == 0: max[0] += 0.0001
  if max[1] - min[1] == 0: max[1] += 0.0001
  if max[2] - min[2] == 0: max[2] += 0.0001
  newList[:, 0] = (lmlist[:, 0] - min[0]) / (1.0 * max[0] - min[0])
  newList[:, 1] = (lmlist[:, 1] - min[1]) / (1.0 * max[1] - min[1])
  newList[:, 2] = (lmlist[:, 2] - min[2]) / (1.0 * max[2] - min[2])
  return newList

for i in range(len(data)):
    data[i] = preprocess(data[i])

data = data.reshape(data.shape[0], 21*3) # including z

translate = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
    ]

model = MLPC(hidden_layer_sizes=(175,), solver="adam", max_iter=500)
model.fit(data, labels)
print(model.score(data, labels))
print(np.mean(cross_val_score(model, data, labels)))
input()
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    if lmlist:
        lmlist = preprocess(np.array(lmlist, dtype=float))
        # if lmlist.shape != (21, 2): #not including z
        if lmlist.shape != (21, 3): #including z
            continue

        lmlist = lmlist.reshape(1, 21*3) # including z
        pred = model.predict(lmlist)[0]
        # letter = newTranslate[pred]
        letter = translate[pred]
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter, (h-20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break