import tensorflow as tf
import os

def load_keras_model(model_path):
    """Load a TensorFlow Keras model with improved error handling."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    if not os.path.isfile(model_path):
        print(f"Error: {model_path} is not a valid file")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except tf.errors.NotFoundError:
        print(f"Error: Model file not found or corrupted: {model_path}")
    except tf.errors.OpError:
        print(f"Error: Cannot access model file: {model_path}")
    except ValueError as e:
        print(f"Error: Invalid model format: {str(e)}")
    except Exception as e:
        print(f"Unexpected error loading Keras model: {str(e)}")
    return None

if __name__ == "__main__":
    keras_model_path = 'Models\Gender_Lastv3_last.h5'
    model = load_keras_model(keras_model_path)
