import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_vs_wolf_model.h5")

model = load_model()

# App title
st.title("🐶 Dog vs 🐺 Wolf Classifier")
st.write("Upload an image and the model will predict whether it is a dog or a wolf.")

# Upload image
uploaded_file = st.file_uploader(
"Choose an image...",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((160, 160))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

try:
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    dog_conf = 1 - prediction
    wolf_conf = prediction
    confidence = max(dog_conf, wolf_conf)

    # Reject uncertain predictions
    if confidence < 0.85:
        st.error(
            "⚠️ The model is not confident this image is a dog or a wolf. Please upload another image."
        )
    else:
        if wolf_conf > dog_conf:
            label = "🐺 Wolf"
        else:
            label = "🐶 Dog"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

except Exception as e:
    st.error(f"Prediction failed: {e}")
