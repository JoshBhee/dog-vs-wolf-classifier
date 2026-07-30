# Dogs vs Wolf Image Classification

## Project Overview
This project is an AI-powered image classification application developed using TensorFlow/Keras and Streamlit. It classifies uploaded images as either **Dog** or **Wolf** using a trained Convolutional Neural Network (CNN).

## Features
- Upload JPG, JPEG, or PNG images.
- Predicts whether the image is a Dog or a Wolf.
- Displays prediction confidence.
- Rejects images that are not confidently classified.

## Technologies Used
- Python
- TensorFlow/Keras
- Streamlit
- NumPy
- Pillow

## Project Structure

Dogs_vs_Wolf/
│── app.py
│── dogs_vs_wolf_model.keras
│── requirements.txt
│── README.md

## How to Run

1. Install the required packages:

pip install -r requirements.txt

2. Start the application:

streamlit run app.py

3. Upload an image and wait for the prediction.

## Model

The model was trained using a publicly available Dogs vs Wolf image dataset and saved in Keras (.keras) format.

## Future Improvements

- Increase dataset size.
- Improve prediction accuracy.
- Support additional animal classes.
