import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("emotion_recognition_mobilenet.keras")

# Define emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

st.title("Facial Expression Recognition App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    #  Debugging: Print shape before preprocessing
    st.write(f"Original Image Shape: {image.shape}")

    #  Ensure 3-channel RGB input for MobileNet
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # If RGBA (4 channels)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    #  Debugging: Print shape after conversion
    st.write(f"Processed Image Shape: {image.shape}")

    # Resize to 48x48 and normalize
    img_resized = cv2.resize(image, (48, 48)) / 255.0

    # Reshape for model input (1, 48, 48, 3) - MobileNet expects 3 channels
    img_resized = np.expand_dims(img_resized, axis=0)

    #  Debugging: Print shape before prediction
    st.write(f"Final Image Shape for Model: {img_resized.shape}")

    # Predict emotion
    predictions = model.predict(img_resized)
    predicted_class = np.argmax(predictions)

    # Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"Predicted Emotion: {EMOTION_LABELS[predicted_class]}")
