from ultralytics import YOLO

# Load the pre-trained YOLOv8 model for face detection
yolo_model = YOLO('Models/best10.pt')

def detect_faces(frame):
    results = yolo_model(frame)
    rects = []
    if results is not None and len(results) > 0:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rects.append((x1, y1, x2, y2))
    return rects

if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rects = detect_faces(frame)
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('YOLO Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
