import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("--- Starting Braille Training Script ---")

# 1. Setup Paths
dataset_path = "data/braille_dataset"

if not os.path.exists(dataset_path):
    print(f"ERROR: The folder '{dataset_path}' was not found!")
    print("Please make sure your images are in: D:\\Braille_Translator\\data\\braille_dataset")
    exit()

print(f"Success: Folder '{dataset_path}' found.")

# 2. Load Dataset
print("Loading images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=32,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print(f"Detected {len(class_names)} characters: {class_names}")

# 3. Preprocessing
normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Build Model
print("Building the Neural Network...")

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train
print("Beginning Training (This may take a few minutes)...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. Save Model
os.makedirs("models", exist_ok=True)
model.save("models/braille_model.h5")

print("--- SUCCESS: Model saved as models/braille_model.h5 ---")
print("Training Complete.")