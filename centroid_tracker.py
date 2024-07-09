import numpy as np
import time
from scipy.spatial import distance as dist
import cv2

class CentroidTracker:
    def __init__(self, maxDisappeared=50, disappeared_time_threshold=2):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.last_detected = {}
        self.maxDisappeared = maxDisappeared
        self.disappeared_time_threshold = disappeared_time_threshold  # seconds

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.last_detected[self.nextObjectID] = time.time()
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.last_detected[objectID]

    def update(self, rects):
        if len(rects) == 0:
            current_time = time.time()
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if (current_time - self.last_detected[objectID]) > self.disappeared_time_threshold:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.last_detected[objectID] = time.time()
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if (time.time() - self.last_detected[objectID]) > self.disappeared_time_threshold:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ct = CentroidTracker()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        rects = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
