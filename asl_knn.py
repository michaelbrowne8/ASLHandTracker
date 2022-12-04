import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from handDetector import handDetector as hd
import cv2
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import mediapipe as mp

labels = np.load('/Users/marilyn/Documents/machineLearning/ASLHandTracker/labelsNew.npy')
data = np.load('/Users/marilyn/Documents/machineLearning/ASLHandTracker/dataNew.npy').astype(float) 


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

data = data.reshape(data.shape[0], 21*3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

'''
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

'''
k = [1, 2, 3, 4, 5, 6]
mean_error_rate = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    print(metrics.accuracy_score(y_train, y_pred))
'''

translate = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
newTranslate = ["A", "B", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
             "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"]
removed = [2, 3, 14, 15, 16, 26, 28]


cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter1 = "nothing"

    if lmlist:
        preprocessedHand = preprocess(np.array(lmlist, dtype=float)).reshape(1, -1)
        pred1 = knn.predict(preprocessedHand)[0]

        letter1 = newTranslate[pred1]

    # Displaying hand and guess on screen
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, letter1, (h-20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
