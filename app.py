import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('monument_recognition_model.h5')


# Load class labels (assuming you have train_data or equivalent to fetch the class indices)
# Here you could either manually define the labels or load them based on your training data.
# Example:
# class_labels = ['Class 1', 'Class 2', 'Class 3']
# Or dynamically using something like this:
class_labels = list(train_data.class_indices.keys())

# Streamlit app header
st.title("Monument Recognition")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class = np.argmax(predictions, axis=1)

    # Map the predicted class index to the corresponding class label
    predicted_label = class_labels[predicted_class[0]]

    # Show the predicted label
    st.write(f"Predicted label: {predicted_label}")
    # Optionally, display the confidence score of the prediction
    confidence_score = np.max(predictions) * 100
    st.write(f"Confidence: {confidence_score:.2f}%")

    # Optionally, show the prediction graph for all classes
    st.write("Prediction Distribution:")
    st.bar_chart(predictions[0])
    
# Additional Features:

# Show some instructions on how to use the app
st.sidebar.title("How to Use the App")
st.sidebar.write(
    "1. Upload an image of a monument (jpg, png, jpeg). \n"
    "2. The model will predict the monument category. \n"
    "3. See the confidence score and distribution chart."
)
