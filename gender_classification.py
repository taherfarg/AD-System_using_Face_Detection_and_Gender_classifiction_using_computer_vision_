import cv2
import numpy as np

def preprocess_image(img):
    """Preprocess image for the TensorFlow Keras model."""
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_gender(face_img, model):
    """Classify gender using the TensorFlow Keras model."""
    processed_img = preprocess_image(face_img)
    prediction = model.predict(processed_img)
    return "male" if prediction[0][0] > 0.5 else "female"

if __name__ == "__main__":
    from load_model import load_keras_model
    model = load_keras_model('Models/Gender_Lastv3_last.h5')
    if model:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = frame[100:200, 100:200]  # Dummy face crop for demonstration
            gender = classify_gender(face, model)
            print(f"Detected gender: {gender}")
            cv2.imshow('Gender Classification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
