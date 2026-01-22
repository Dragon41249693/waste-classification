import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
MODEL_PATH = "waste_classifier_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Test on a few images from test dir
test_dir = "test"
for category in class_labels:
    cat_dir = os.path.join(test_dir, category)
    if os.path.exists(cat_dir):
        files = os.listdir(cat_dir)[:3]  # First 3 images
        for f in files:
            img_path = os.path.join(cat_dir, f)
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224))
                x = image.img_to_array(img_resized)
                x = np.expand_dims(x, axis=0) / 255.0
                preds = model.predict(x)
                confidence_scores = preds[0]
                predicted_idx = np.argmax(confidence_scores)
                predicted_class = class_labels[predicted_idx]
                confidence = confidence_scores[predicted_idx]
                print(f"True: {category}, Predicted: {predicted_class}, Confidence: {confidence:.2f}")
                print(f"Scores: {confidence_scores}")
            except Exception as e:
                print(f"Error on {img_path}: {e}")