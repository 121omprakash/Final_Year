import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('monument_recognition_model.h5')

# Hardcode the class labels
class_labels = [
    'Aga Khan Palace', 'Ajanta Caves', 'Class 3', 'Ajmeri Gate Delhi', 
    'Albert Hall Museum', 'Badrinath Temple', 'Bara Imambara', 
    'Barsi Gate Hansi Old', 'Basilica of Bom Jesus', 'Bekal', 
    'Bharat Mata Mandir Haridwar', 'Bhoramdev Mandir', 'Buddha Temple', 
    'Bidar Fort', 'Brihadeshwara Temple', 'Buland Darwaza', 
    'Byzantine Architecture', 'Cathedral', 'Champaner', 
    'Chandi Devi Mandir Haridwar', 'Chandigarh College of Architecture', 
    'Chapora Fort', 'Charminar', 'Cheese', 
    'Chhattisgarh Ke Saat Ajube', 'Chhatrapati Shivaji Statue', 
    'Chhatrapati Shivaji Terminus', 'Chittorgarh', 
    'Chittorgarh Padmini Lake Palace', 'City Palace', 'Daman', 
    'Dhamek Stupa', 'Diu', 'Diu Museum', 'Dome', 
    'Dubdi Monastery Yuksom Sikkim', 'Falaknuma Palace', 
    'Fatehpur Sikri', 'Fatehpur Sikri Fort', 'Ford Auguda', 
    'Fortification', 'Gol Ghar', 'Golden Temple', 'Hampi', 
    'Hawa Mahal', 'Hidimbi Devi Temple', 'Hindu Temple', 
    'Hoshang Shah Tomb', 'India Gate', 'Isarlat Sargasooli'
]  # Replace with your actual class names

# Set up the title and description
st.title("Monument Recognition")
st.write("Upload an image or use your camera to capture one, and the model will predict the monument category.")

# Option to upload an image or capture using camera
image_option = st.radio("Choose Image Source", ("Upload an Image", "Use Camera"))

if image_option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image
        img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize to model's input size
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
elif image_option == "Use Camera":
    captured_image = st.camera_input("Capture an image")

    if captured_image is not None:
        # Display the captured image
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        img = image.load_img(captured_image, target_size=(224, 224))  # Resize to model's input size

# If an image is uploaded or captured
if uploaded_file is not None or captured_image is not None:
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
    st.bar_chart(predictions[0])

# Additional Features:
# Show some instructions on how to use the app
st.sidebar.title("How to Use the App")
st.sidebar.write(
    "1 . Upload an image of a monument (jpg, png, jpeg) or capture an image using your camera. \n"
    "2. The model will predict the monument category. \n"
    "3. See the confidence score and distribution chart."
)
