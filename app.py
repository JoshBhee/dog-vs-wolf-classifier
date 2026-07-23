import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image
import numpy as np

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    classifier = tf.keras.models.load_model("dog_vs_wolf_model.h5")
    mobilenet = MobileNetV2(weights="imagenet")
    return classifier, mobilenet

model, mobilenet = load_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐶 Dog vs 🐺 Wolf Classifier")
st.write(
    "Upload an image. The app first checks whether it is a dog or wolf. "
    "If not, it rejects the image."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ======================================================
    # Step 1: MobileNet validation
    # ======================================================

    mobile_img = image.resize((224, 224))
    mobile_array = np.array(mobile_img, dtype=np.float32)
    mobile_array = np.expand_dims(mobile_array, axis=0)
    mobile_array = preprocess_input(mobile_array)

    preds = mobilenet.predict(mobile_array, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    labels = [item[1].lower() for item in decoded]

    allowed_keywords = [
        "dog",
        "wolf",
        "husky",
        "malamute",
        "retriever",
        "labrador",
        "shepherd",
        "terrier",
        "beagle",
        "poodle",
        "chihuahua",
        "spaniel",
        "collie",
        "rottweiler",
        "doberman",
        "boxer",
        "greyhound",
        "corgi",
        "samoyed",
        "dingo",
        "wild_dog",
        "red_wolf",
        "timber_wolf",
        "white_wolf",
    ]

    is_valid = any(
        any(keyword in label for keyword in allowed_keywords)
        for label in labels
    )

    if not is_valid:
        st.error("❌ This image is neither a dog nor a wolf.")
        st.stop()

    # ======================================================
    # Step 2: Your custom classifier
    # ======================================================

    img = image.resize((160, 160))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "🐺 Wolf"
        confidence = prediction * 100
    else:
        label = "🐶 Dog"
        confidence = (1 - prediction) * 100

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")

    with st.expander("MobileNet detected"):
        for _, name, score in decoded:
            st.write(f"{name}: {score*100:.2f}%")
