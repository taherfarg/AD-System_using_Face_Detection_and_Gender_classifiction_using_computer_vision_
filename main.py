import threading
import os
import shutil
import cv2
import time
from datetime import datetime
import mediapipe as mp
import requests
from load_model import load_keras_model
from yolo_face_detection import detect_faces
from gender_classification import classify_gender
from centroid_tracker import CentroidTracker

# Endpoint details
endpoint_url = "https://ads-track.smaster.live/api.php"
key = "kOEjOeaoL7BmgxC6PCM5GZsetaxq698hzgHv81Kd6XxfTsOM2W"

stop_event = threading.Event()

def play_video(video_path):
    """Play video using OpenCV with improved error handling."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            return

        cv2.namedWindow('Ad', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Ad', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Ad', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error playing video {video_path}: {str(e)}")
        try:
            cv2.destroyAllWindows()
        except:
            pass

def send_data_to_endpoint(gender):
    """Send data to the specified endpoint with improved error handling."""
    try:
        url = endpoint_url
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            "key": key,
            "gender": gender,
            "datetime": timestamp
        }

        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("Response from server:", response_data)
            except ValueError:
                print("Response from server (non-JSON):", response.text)
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Network error sending data to endpoint: {str(e)}")
    except Exception as e:
        print(f"Unexpected error sending data: {str(e)}")

def process_video_stream():
    """Process video stream to detect faces and classify gender with improved error handling."""
    model_path = 'Models/Gender_Lastv3_last.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    model = load_keras_model(model_path)
    if model is None:
        print("TensorFlow Keras model is not loaded. Exiting...")
        return

    # Check if ad video files exist
    men_ad_path = 'AD_Videos/MenAD.mp4'
    women_ad_path = 'AD_Videos/womanAD.mp4'
    family_ad_path = 'AD_Videos/Familyad.mp4'

    for video_path, video_type in [(men_ad_path, "Men"), (women_ad_path, "Women"), (family_ad_path, "Family")]:
        if not os.path.exists(video_path):
            print(f"Warning: {video_type} advertisement not found: {video_path}")

    try:
        cap = cv2.VideoCapture(0)  # Capture video from webcam
        if not cap.isOpened():
            print("Error: Could not open webcam. Trying alternative camera indices...")
            # Try different camera indices
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Successfully opened camera at index {i}")
                    break
            else:
                print("Error: Could not open any webcam. Please check your camera connection.")
                return
    except Exception as e:
        print(f"Error initializing video capture: {str(e)}")
        return

    men_ad_path = 'AD_Videos/MenAD.mp4' 
    women_ad_path = 'AD_Videos/womanAD.mp4'  
    family_ad_path = 'AD_Videos/Familyad.mp4'

    gender_timer = {'male': 0, 'female': 0, 'both': 0}
    gender_detected = {'male': False, 'female': False}
    last_time = time.time()
    unique_ids = set()
    gender_per_id = {}

    ct = CentroidTracker()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Detect faces with error handling
            try:
                rects = detect_faces(frame)
            except Exception as e:
                print(f"Error in face detection: {str(e)}")
                rects = []

            if rects:
                for (x1, y1, x2, y2) in rects:
                    try:
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0 and model is not None:
                            gender = classify_gender(face, model)
                            gender_detected[gender] = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")
                        continue

        objects = ct.update(rects)

        if len(objects) == 0:
            if unique_ids:
                unique_ids.clear()
                print("No objects detected for 2 seconds, resetting IDs.")
                gender_per_id.clear()
        else:
            for (objectID, centroid) in objects.items():
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                if objectID not in unique_ids:
                    unique_ids.add(objectID)
                    gender_per_id[objectID] = gender

        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time

        if gender_detected['male'] and gender_detected['female']:
            gender_timer['both'] += elapsed_time
            gender_timer['male'] = 0
            gender_timer['female'] = 0
        elif gender_detected['male']:
            gender_timer['male'] += elapsed_time
            gender_timer['female'] = 0
            gender_timer['both'] = 0
        elif gender_detected['female']:
            gender_timer['female'] += elapsed_time
            gender_timer['male'] = 0
            gender_timer['both'] = 0
        else:
            gender_timer['male'] = 0
            gender_timer['female'] = 0
            gender_timer['both'] = 0

        if gender_timer['male'] >= 2:
            play_video(men_ad_path)
            send_data_to_endpoint('male')
            gender_timer['male'] = 0
            gender_timer['female'] = 0
            gender_timer['both'] = 0
        elif gender_timer['female'] >= 2:
            play_video(women_ad_path)
            send_data_to_endpoint('female')
            gender_timer['female'] = 0
            gender_timer['male'] = 0
            gender_timer['both'] = 0
        elif gender_timer['both'] >= 2:
            play_video(family_ad_path)
            send_data_to_endpoint('male')
            send_data_to_endpoint('female')
            gender_timer['both'] = 0
            gender_timer['male'] = 0
            gender_timer['female'] = 0

        gender_detected['male'] = False
        gender_detected['female'] = False

        cv2.putText(frame, f'Count: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, current_time_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # MediaPipe face mesh processing with error handling
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        except Exception as e:
                            print(f"Error drawing face landmarks: {str(e)}")
            except Exception as e:
                print(f"Error in MediaPipe processing: {str(e)}")

            cv2.imshow('Gender Classification and Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("Received keyboard interrupt, shutting down gracefully...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {str(e)}")
            continue

    # Cleanup
    try:
        cap.release()
        cv2.destroyAllWindows()
        print("Video stream processing stopped.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def stop_video_stream():
    stop_event.set()
