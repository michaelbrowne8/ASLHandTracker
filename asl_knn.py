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

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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

'''
cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    
    if lmlist:
        preprocessedHand = preprocess(np.array(lmlist, dtype=float).reshape(1, 21*3))
        pred = knn.predict(preprocessedHand)[0]
        letter = newTranslate[pred]
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime=cTime
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, letter, (h-20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
'''
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                #coords = scaler.transform([coords])
                
                # Alternative for dataset using z coordinates.
                # Z coordinates is not recommended, since you need to adjust your distance from camera.
#                 coords = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in coords]).flatten())
                
                predicted = knn.predict(coords)

            # Get status box
            cv2.rectangle(image, (0,0), (100, 60), (245, 90, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS'
                        , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted[0])
                        , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)

        # Press esc to close webcam
        if cv2.waitKey(5) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
