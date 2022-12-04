import cv2
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands   
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, im, draw=True):
        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(im, handLms, self.mpHands.HAND_CONNECTIONS)
        return im

    def findPosition(self, im, handNo=0, draw=True):
        lmList = []
        h, w, c = im.shape
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for lm in myHand.landmark:
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([lm.x, lm.y, lm.z])
                if draw: cv2.circle(im, (cx, cy), 12, (203, 169, 69), cv2.FILLED)
        return lmList

    def newHands(self, im, draw=True):
        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imRGB)
        blank = np.zeros(im.shape)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(blank, handLms, self.mpHands.HAND_CONNECTIONS)
        return blank

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=4)
    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape
        frame = frame[:, (w-h)//2:w-(w-h)//2, :]
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)

        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def otherMain():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=4)
    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape
        frame = frame[:, (w-h)//2:w-(w-h)//2, :]
        new = detector.newHands(frame)
        # frame = detector.findHands(frame)

        # cTime = time.time()
        # fps = int(1/(cTime-pTime))
        # pTime = cTime
        # frame = cv2.flip(frame, 1)
        new = cv2.flip(new, 1)
        # cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow('frame', new)
        # cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # main()
    otherMain()