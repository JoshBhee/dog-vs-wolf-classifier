import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

Load the trained model
@st.cache_resource
def load_model():
return tf.keras.models.load_model("dog_vs_wolf_model.h5")

model = load_model()

App title
st.title("🐶 Dog vs 🐺 Wolf Classifier")
st.write("Upload an image and the model will predict whether it is a dog or a wolf.")

Upload image
uploaded_file = st.file_uploader(
"Choose an image...",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
# Open image
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_container_width=True)

# Resize and preprocess
img = image.resize((160, 160))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

try:
    # Make prediction
    prediction = float(model.predict(img_array, verbose=0)[0][0])

    # Reject uncertain predictions
    confidence_gap = abs(prediction - 0.5)

    if confidence_gap < 0.15:
        st.error("⚠️ This does not appear to be a dog or a wolf. Please upload a clearer image.")
    else:
        if prediction > 0.5:
            label = "🐺 Wolf"
            confidence = prediction * 100
        else:
            label = "🐶 Dog"
            confidence = (1 - prediction) * 100

        st.success(f"Prediction: {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

except Exception as e:
    st.error(f"Prediction failed: {e}")
