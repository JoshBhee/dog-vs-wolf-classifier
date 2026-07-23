import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
	return tf.keras.models.load_model('dog_vs_wolf_model.h5')

model = load_model()

st.title("Dog vs Wolf Classifier")
st.write("Upload an image and the model will predict whether it's a dog or a wolf.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	image = Image.open(uploaded_file).convert('RGB')
	st.image(image, caption='Uploaded Image', use_container_width=True)

	img_size = (160, 160)
	img = image.resize(img_size)
	img_array = np.array(img) / 255.0
	img_array = np.expand_dims(img_array, axis=0)

	prediction = model.predict(img_array)[0][0]

	if prediction > 0.5:
    		label = "Wolf"
    		confidence = prediction * 100
	else:
    		label = "Dog"
    		confidence = (1 - prediction) * 100

	st.subheader(f"Prediction: {label}")
	st.write(f"Confidence: {confidence:.2f}%")
