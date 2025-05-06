import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


# Path to save the model
model_path = "urban_rural_model.h5"

# Download only if not already present
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1iPBWUv6CuViDjyob5LIYk1B73XgHxOK5"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# App Title and Description
st.title("Rural or Urban Image Classifier")

st.image(Image.open('urban-rural.png'))

st.write("""
Upload an image, and the model will classify it as **urban** or **rural** using ResNet50 and transfer learning.
""")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button('Predict'):
        # 1. Preprocess the image
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Specific to ResNet50

        # 2. Predict
        prediction = model.predict(img_array)[0][0]
        predicted_class = 'Urban' if prediction >= 0.5 else 'Rural'

        # 4. Display result
        st.subheader("Prediction:")
        st.write(f"The image is classified as: **{predicted_class}**")
        #st.write(f"Model confidence: `{prediction:.4f}` (1 = Rural, 0 = Urban)")

