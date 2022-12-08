import numpy as np
from handDetector import handDetector as hd
import cv2
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


'''
At the very bottom of the file is the hand detector using the webcam
This one we can actually make into a separate file so that it can just be 
    imported and ran
But it should show the tracked hand and the letter it thinks it is

Press 'Q' to quit
'''

labels = np.load('labelsNew.npy') # 16502 labels
data = np.load('dataNew.npy').astype(float)    # 16502 preprocessed images 
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
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ""
    ]

labels = np.where(labels == 27, 26, labels)
labels = np.where(labels == 28, 26, labels)
'''
model = MLPC(hidden_layer_sizes=(175,), max_iter=500)
model.fit(data, labels)
print(model.score(data, labels))
print(np.mean(cross_val_score(model, data, labels)))
answer = input("do you want to save? ")
if answer == "y":
    pickle.dump(model, open("nnWeights.sav", "wb"))
'''
# loading model
model = pickle.load(open('nnWeights.sav', 'rb'))


cm = confusion_matrix(model.predict(data), labels)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)

cm_display.plot()
plt.show()
assert(1==2)

cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = ""
    if lmlist:
        lmlist = preprocess(np.array(lmlist, dtype=float))
        if lmlist.shape != (21, 3):
            continue

        lmlist = lmlist.reshape(1, 21*3)
        pred = model.predict(lmlist)[0]
        letter = translate[pred]
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, letter, (10, 80), 2, 3, (203, 169, 69), 3) #BGR hand color

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break