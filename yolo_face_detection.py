from ultralytics import YOLO
import os

def load_yolo_model(model_path='Models/best10.pt'):
    """Load YOLO model with error handling."""
    if not os.path.exists(model_path):
        print(f"Error: YOLO model file not found at {model_path}")
        return None

    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return None

# Load the pre-trained YOLOv8 model for face detection
yolo_model = load_yolo_model()

def detect_faces(frame):
    """Detect faces in frame with error handling."""
    if yolo_model is None:
        print("Error: YOLO model not loaded")
        return []

    try:
        results = yolo_model(frame, conf=0.5)  # Add confidence threshold
        rects = []
        if results is not None and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Ensure coordinates are within frame bounds
                            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                                rects.append((x1, y1, x2, y2))
                        except (ValueError, IndexError) as e:
                            print(f"Error processing bounding box: {str(e)}")
                            continue
        return rects
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return []

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
