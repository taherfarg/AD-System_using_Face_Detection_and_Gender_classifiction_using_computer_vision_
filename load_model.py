import tensorflow as tf

def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

if __name__ == "__main__":
    keras_model_path = 'Models\Gender_Lastv3_last.h5'
    model = load_keras_model(keras_model_path)
