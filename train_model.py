import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = "."
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Check if directories exist
for dir_path in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist. Please ensure the dataset directories are set up correctly.")

# -------------------------------
# PARAMETERS
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# -------------------------------
# DATA GENERATORS (WITH AUGMENTATION)
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -------------------------------
# TRANSFER LEARNING MODEL
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
callbacks = [
    ReduceLROnPlateau(monitor='loss', patience=2, factor=0.3, verbose=1)
]

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_gen,
    epochs=10,  # Train only the top layers for a few epochs first
    callbacks=callbacks
)

# -------------------------------
# FINE-TUNE
# -------------------------------
# Unfreeze the top layers of the model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Continue training
history_fine = model.fit(
    train_gen,
    epochs=EPOCHS,  # Continue training for more epochs
    initial_epoch=history.epoch[-1],
    callbacks=callbacks
)

# -------------------------------
# TEST EVALUATION
# -------------------------------
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# -------------------------------
# SAVE MODEL
# -------------------------------
MODEL_PATH = "waste_classifier_cnn.h5"
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")