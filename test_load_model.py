import tensorflow as tf

print("TensorFlow version:", tf.__version__)

model_path = "models/braille_model.h5"

try:
    model = tf.keras.models.load_model(model_path)
    print("Loaded OK. Summary:")
    model.summary()
except Exception as e:
    print("FAILED to load:", e)
