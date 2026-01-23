# AI Waste Classification System
# Modular design recommended for production:
# - preprocessing.py: Image preprocessing utilities
# - explainability.py: Grad-CAM and model interpretation
# - impact.py: Environmental impact calculations
# - model/: Directory for saved models and metadata

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
from collections import Counter

# -------------------------------
# Constants
# -------------------------------
CONFIDENCE_THRESHOLD_LOW = 0.2
CONFIDENCE_THRESHOLD_MEDIUM = 0.4
CONFIDENCE_THRESHOLD_HIGH = 0.7
AMBIGUITY_THRESHOLD = 0.15
IMAGE_SIZE = (224, 224)
MIN_IMAGE_SIZE = 150
MAX_FILES = 10

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "waste_classifier_cnn.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
except Exception as e:
    st.error(f"❌ Failed to load model: {e}. Please ensure '{MODEL_PATH}' exists.")
    st.stop()

# Class labels (MUST match training order)
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Model info (update as needed)
MODEL_INFO = {
    "model_type": "CNN (Transfer Learning)",
    "input_size": "224×224 RGB",
    "classes": 6,
    "evaluation": "Test accuracy reported in training notebook"
}

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="🌍 AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Sidebar with model info
st.sidebar.markdown("### Model Info")
st.sidebar.write(f"Type: {MODEL_INFO['model_type']}")
st.sidebar.write(f"Input: {MODEL_INFO['input_size']}")
st.sidebar.write(f"Classes: {MODEL_INFO['classes']}")
st.sidebar.write(f"Evaluation: {MODEL_INFO['evaluation']}")

st.title("🌍 AI-Based Waste Classification System ♻️")
st.markdown(
    """
This application uses a **Convolutional Neural Network (CNN)** to classify waste images
and provide **recycling guidance & environmental impact awareness** 🌱
"""
)


st.divider()

# -------------------------------
# Multi-image upload
# -------------------------------
uploaded_files = st.file_uploader(
    f"📤 Upload one or more waste images (max {MAX_FILES})",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.error(f"❌ Too many files. Please upload at most {MAX_FILES} images.")
    uploaded_files = uploaded_files[:MAX_FILES]

if uploaded_files:
    predictions_summary = []

    # Process each image individually
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"🖼️ Image: {uploaded_file.name}")

        try:
            # Load & show image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, width=300)

            # Image quality notice
            if img.size[0] < MIN_IMAGE_SIZE or img.size[1] < MIN_IMAGE_SIZE:
                st.warning("⚠️ Image resolution is low — prediction accuracy may be affected")

            # Preprocess
            img_resized = img.resize(IMAGE_SIZE)
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0) / 255.0

            # Prediction
            preds = model.predict(x)
            confidence_scores = preds[0]

            # Confidence threshold auto-rejection
            if np.max(confidence_scores) < CONFIDENCE_THRESHOLD_LOW:
                st.error(
                    "🚫 **Prediction rejected**\n\n"
                    "The model confidence is too low for a reliable decision."
                )
                continue

            class_idx = np.argmax(confidence_scores)
            predicted_class = class_labels[class_idx]
            predictions_summary.append(predicted_class)

            # Top-3 predictions
            top_indices = np.argsort(confidence_scores)[::-1][:3]

            st.subheader("📊 Top Predictions")
            for j in top_indices:
                st.write(f"**{class_labels[j].capitalize()}** — {confidence_scores[j]*100:.2f}%")

            # Visual confidence chart
            chart_data = {
                "Class": [class_labels[j].capitalize() for j in top_indices],
                "Confidence (%)": [confidence_scores[j] * 100 for j in top_indices]
            }
            st.bar_chart(chart_data, x="Class", y="Confidence (%)", height=200)

            # Material Confusion Warning
            top_probs = np.sort(confidence_scores)[-3:]
            if top_probs[-1] - top_probs[-2] < AMBIGUITY_THRESHOLD:
                st.warning(
                    "⚠️ **Material ambiguity detected**\n\n"
                    "This item shares visual features with multiple waste categories. "
                    "Consider checking labels or consulting local recycling guidelines."
                )

            confidence = float(confidence_scores[class_idx])

            # -------------------------------
            # Prediction result
            # -------------------------------
            st.markdown(
                f"### 🏷️ Prediction: **{predicted_class.upper()}**"
            )
            st.markdown(
                f"**Confidence:** {confidence*100:.2f}%"
            )

            # -------------------------------
            # Confidence-aware decision
            # -------------------------------
            if confidence >= CONFIDENCE_THRESHOLD_HIGH:
                st.success("✅ High confidence prediction")
            elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
                st.warning("⚠️ Medium confidence — manual verification recommended")
            else:
                st.error("❌ Low confidence — model unsure")

            # Confidence bar
            st.progress(confidence)

            st.markdown("### 🧠 Model Interpretation")

            if confidence >= CONFIDENCE_THRESHOLD_HIGH:
                st.write(
                    "The model is **highly confident**, meaning the image features strongly match "
                    "patterns learned during training. The AI has identified clear visual characteristics "
                    "typical of this waste category."
                )
            elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
                st.write(
                    "The model shows **moderate confidence**, possibly due to overlapping features "
                    "between waste categories. This is common for materials with similar appearances."
                )
            else:
                st.write(
                    "The model is **uncertain**, which may be caused by poor image quality, unusual angles, "
                    "or uncommon waste material not well-represented in the training data."
                )

            # Class-specific explanations
            class_explanations = {
                "cardboard": "Cardboard is identified by its fibrous texture, brown color, and rectangular shapes. It may be confused with paper due to similar materials.",
                "glass": "Glass is recognized by its transparent/reflective properties and smooth, curved surfaces. Clear plastic bottles can sometimes be mistaken for glass.",
                "metal": "Metal objects are detected by their shiny, reflective surfaces and rigid shapes. Aluminum foil or cans are common examples.",
                "paper": "Paper is characterized by its thin, flexible nature and printed text/images. It can be confused with cardboard or certain plastics.",
                "plastic": "Plastic is identified by its glossy appearance, molded shapes, and often colorful designs. Different plastics may look similar to the AI.",
                "trash": "Non-recyclable waste includes items that don't fit other categories, often organic or composite materials."
            }

            if predicted_class in class_explanations:
                st.markdown("#### 🔍 Why This Prediction?")
                st.write(class_explanations[predicted_class])

            # TODO: Add Grad-CAM visualization here for visual explainability
            # This would show heatmaps of image regions that influenced the prediction
            # Implementation would require tf-explain or custom Grad-CAM function

            # -------------------------------
            # Environmental impact messages
            # -------------------------------
            recyclable = ["cardboard", "glass", "metal", "paper", "plastic"]
            recycling_tips = {
                "cardboard": "• Flatten boxes to save space\n• Remove tape, labels, and staples\n• Keep dry to prevent mold\n• Stack neatly for collection 📦",
                "glass": "• Rinse thoroughly\n• Remove caps and lids\n• Sort by color if required\n• Handle carefully to avoid breakage 🍾",
                "metal": "• Rinse cans and containers\n• Crush aluminum cans\n• Remove paper labels\n• Keep ferrous and non-ferrous separate 🔩",
                "paper": "• Remove food residue and grease\n• Flatten boxes and cartons\n• Remove plastic windows from envelopes\n• Keep dry and clean 📄",
                "plastic": "• Check resin identification code\n• Rinse bottles and containers\n• Remove caps and labels\n• Crush to save space 🧴"
            }
            impact_stats = {
                "cardboard": "Recycling 1 ton saves ~17 trees 🌳",
                "glass": "100% recyclable without quality loss ♻️",
                "metal": "Saves 95% of energy vs mining ⚡",
                "paper": "Recycling 1 ton saves ~17 trees 🌳",
                "plastic": "Reduces ocean pollution 🐋",
                "trash": "Non-recyclable - proper disposal needed"
            }

            if predicted_class in recyclable:
                st.success(
                    "♻️ **This item is recyclable!**\n\n"
                    "🌱 Recycling helps reduce pollution and conserve natural resources."
                )

                if predicted_class in impact_stats:
                    st.markdown(f"🌍 **Environmental Impact:** {impact_stats[predicted_class]}")

                st.markdown("🌍 **Did you know?**")
                st.write(
                    "Improper waste disposal contributes to land, water, and air pollution. "
                    "AI-assisted waste sorting can significantly increase recycling efficiency."
                )

                if predicted_class in recycling_tips:
                    st.markdown("### 📝 Recycling Instructions")
                    st.markdown(recycling_tips[predicted_class])

            else:
                st.error(
                    "🗑️ **This item is non-recyclable**\n\n"
                    "⚠️ Dispose responsibly to protect the environment."
                )
                if predicted_class in impact_stats:
                    st.markdown(f"🌍 **Environmental Impact:** {impact_stats[predicted_class]}")

        except Exception as e:
            st.error(f"❌ Error processing image {uploaded_file.name}: {e}")


else:
    st.info("ℹ️ Upload one or more images to start classification.")