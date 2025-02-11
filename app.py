import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('monument_recognition_model.h5')

# Hardcode the class labels
class_labels = [
    'Aga Khan Palace', 'Badrinath Temple', 'Bekal', 'Bhudha Temple', 
    'Brihadeshwara Temple', 'Cathederal', 'Champaner', 
    'Chandi Devi mandir hariwar', 'Cheese', 'Chhatrapati Shivaji terminus', 
    'Chittorgarh Padmini Lake Palace', 'Daman', 'Diu Museum', 
    'Fatehpur Sikri Fort', 'Hampi', 'Hoshang Shah Tomb', 
    'India Gate', 'Isarlat Sargasooli', 'ajanta caves', 
    'ajmeri gate delhi', 'albert hall museum', 'bara imambara', 
    'barsi gate hansi old', 'basilica of bom jesus', 
    'bharat mata mandir haridwar', 'bhoramdev mandir', 
    'bidar fort', 'buland darwaza', 'byzantine architecture', 
    'chandigarh college of architecture', 'chapora fort', 
    'charminar', 'chhatisgarh ke saat ajube', 
    'chhatrapati shivaji statue', 'chittorgarh', 
    'city palace', 'dhamek stupa', 'diu', 
    'dome', 'dubdi monastery yuksom sikkim', 
    'falaknuma palace', 'fatehpur sikri', 'ford Auguda', 
    'fortification', 'gol ghar', 'golden temple', 
    'hawa mahal', 'hidimbi devi temple', 'hindu temple'
]  # Replace with your actual class names

# Set up the title and description
st.title("Monument Recognition")
st.write("Upload an image or use your camera to capture one, and the model will predict the monument category.")

# Initialize variables
uploaded_file = None
captured_image = None
img = None  # Initialize img variable

# Option to upload an image or capture using camera
image_option = st.radio("Choose Image Source", ("Upload an Image", "Use Camera"))

if image_option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image
        img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize to model's input size
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
elif image_option == "Use Camera":
    captured_image = st.camera_input("Capture an image")

    if captured_image is not None:
        # Display the captured image
        img = image.load_img(captured_image, target_size=(224, 224))  # Resize to model's input size
        st.image(captured_image, caption="Captured Image", use_container_width=True)

# If an image is uploaded or captured
if img is not None:
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)

    # Map the predicted class index back to the corresponding class label
    predicted_label = class_labels[predicted_class[0]]
    st.write(f"Predicted label: {predicted_label}")

    # Optionally, display the confidence score of the prediction
    confidence_score = np.max(predictions) * 100
    st.write(f"Confidence: {confidence_score:.2f}%")

    # Optionally, show the prediction graph for all classes
    st.write("Prediction Distribution:")
    st.bar_chart(predictions[1])

