import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
from collections import Counter

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "waste_classifier_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Class labels (MUST match training order)
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="🌍 AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

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
    "📤 Upload one or more waste images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    predictions_summary = []
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"🖼️ Image: {uploaded_file.name}")

        try:
            # Load & show image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, width=300)

            # Image quality notice
            if img.size[0] < 150 or img.size[1] < 150:
                st.warning("⚠️ Image resolution is low — prediction accuracy may be affected")

            # Preprocess
            img_resized = img.resize((224, 224))
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0) / 255.0

            # Prediction
            preds = model.predict(x)
            confidence_scores = preds[0]
            
            # Confidence threshold auto-rejection
            if np.max(confidence_scores) < 0.35:
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
            for i in top_indices:
                st.write(f"**{class_labels[i].capitalize()}** — {confidence_scores[i]*100:.2f}%")

            # Material Confusion Warning
            top_probs = np.sort(confidence_scores)[-3:]
            if top_probs[-1] - top_probs[-2] < 0.15:
                st.warning(
                    "⚠️ **Material ambiguity detected**\n\n"
                    "This item shares visual features with multiple waste categories. "
                    "Consider checking labels or consulting local recycling guidelines."
                )

            predicted_class = class_labels[class_idx]
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
            if confidence >= 0.7:
                st.success("✅ High confidence prediction")
            elif confidence >= 0.4:
                st.warning("⚠️ Medium confidence — manual verification recommended")
            else:
                st.error("❌ Low confidence — model unsure")

            # Confidence bar (FIXED float issue)
            st.progress(float(confidence))

            st.markdown("### 🧠 Model Interpretation")

            if confidence >= 0.7:
                st.write(
                    "The model is **highly confident**, meaning the image features strongly match "
                    "patterns learned during training."
                )
            elif confidence >= 0.4:
                st.write(
                    "The model shows **moderate confidence**, possibly due to overlapping features "
                    "between waste categories."
                )
            else:
                st.write(
                    "The model is **uncertain**, which may be caused by poor image quality or uncommon waste material."
                )

            # -------------------------------
            # Environmental impact messages
            # -------------------------------
            recyclable = ["cardboard", "glass", "metal", "paper", "plastic"]
            recycling_tips = {
                "cardboard": "Flatten boxes, keep dry 📦",
                "glass": "Remove caps, avoid breaking 🍾",
                "metal": "Clean cans before disposal 🔩",
                "paper": "Remove food residue, flatten before recycling 📄",
                "plastic": "Check resin code, rinse containers 🧴"
            }

            if predicted_class in recyclable:
                st.success(
                    "♻️ **This item is recyclable!**\n\n"
                    "🌱 Recycling helps reduce pollution and conserve natural resources."
                )

                st.markdown("🌍 **Did you know?**")
                st.write(
                    "Improper waste disposal contributes to land, water, and air pollution. "
                    "AI-assisted waste sorting can significantly increase recycling efficiency."
                )

                if predicted_class in recycling_tips:
                    st.markdown(f"### 📝 How to recycle this item\n{recycling_tips[predicted_class]}")

            else:
                st.error(
                    "🗑️ **This item is non-recyclable**\n\n"
                    "⚠️ Dispose responsibly to protect the environment."
                )


        except Exception as e:
            st.error(f"❌ Error processing image: {e}")


else:
    st.info("ℹ️ Upload one or more images to start classification.")