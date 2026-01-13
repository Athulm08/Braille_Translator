import os
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = "data/braille_dataset"  # your dataset folder

if not os.path.exists(dataset_path):
    print(f"ERROR: {dataset_path} not found!")
    exit()

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=32,
    color_mode="grayscale",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32,
    color_mode="grayscale",
)

class_names = train_ds.class_names
print("Mapping:", class_names)

def prepare(img, label):
    img = tf.cast(img, tf.float32)
    img = 255.0 - img   # invert
    img = img / 255.0   # normalize
    return img, label

train_ds = train_ds.map(prepare)
val_ds = val_ds.map(prepare)

model = models.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=15)

os.makedirs("models", exist_ok=True)
model.save("models/braille_model.h5")
print("--- Model Saved Successfully ---")
