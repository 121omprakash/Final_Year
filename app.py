import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('monument_recognition_model.h5')

# Set up the title and description
st.title("Monument Recognition")
st.write("Upload an image and the model will predict the monument category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If the user uploads an image
if uploaded_file is not None:
    # Open the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    
    # Map the predicted class index back to the corresponding class label
    class_labels = list(model.input_shape[1])
    predicted_label = class_labels[predicted_class[0]]
    
    # Display the prediction result
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
