import cv2
from handDetector import handDetector as hd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# This Preprocess function makes points relative to highest/lowest point on hand instead of location on screen
# Could look into others that may work better, but this work relatively well
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
  return newList.flatten()

# Training model
data = np.load("dataNew.npy")
targets = np.load("labelsNew.npy")
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
           "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

preprocessedData = np.array([preprocess(datum) for datum in data])
model1 = DecisionTreeClassifier()
model1 = model1.fit(preprocessedData, targets)
model2 = LogisticRegression()
model2 = model2.fit(preprocessedData, targets)
model3 = GaussianNB()
model3 = model3.fit(preprocessedData, targets)
model4 = MLPClassifier()
model4 = model4.fit(preprocessedData, targets)


print("Cross Val Score 1", cross_val_score(model1, preprocessedData, targets))
print("Cross Val Score 2", cross_val_score(model2, preprocessedData, targets))
print("Cross Val Score 3", cross_val_score(model3, preprocessedData, targets))
print("Cross Val Score 4", cross_val_score(model4, preprocessedData, targets))


# Live predicting
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter1 = "nothing"
    letter2 = "nothing"
    letter3 = "nothing"
    letter4 = "nothing"

    
    if lmlist:
        preprocessedHand = preprocess(np.array(lmlist, dtype=float)).reshape(1, -1)
        pred1 = model1.predict(preprocessedHand)[0]
        pred2 = model2.predict(preprocessedHand)[0]
        pred3 = model3.predict(preprocessedHand)[0]
        pred4 = model4.predict(preprocessedHand)[0]

        letter1 = classes[pred1] 
        letter2 = classes[pred2] 
        letter3 = classes[pred3] 
        letter4 = classes[pred4] 

        


    # Displaying hand and guess on screen
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, letter1, (h-20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter2, (h-20, 110),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter3, (h-20, 150),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter4, (h-20, 190),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
