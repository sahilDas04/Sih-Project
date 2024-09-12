import streamlit as st
import cv2
import numpy as np
import datetime
import random
from opencage.geocoder import OpenCageGeocode
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore


API_KEY = '7c2e1bab2c41498191953e3110fdee4a'
geocoder = OpenCageGeocode(API_KEY)


MODEL_PATH = 'swachhta_model_improved.h5'
model = load_model(MODEL_PATH)


def get_geolocation():
    locations = ["New Delhi, India", "Mumbai, India", "Chennai, India", "Kolkata, India"]
    location_name = random.choice(locations)
    
    result = geocoder.geocode(location_name)
    if result and len(result) > 0:
        lat = result[0]['geometry']['lat']
        lon = result[0]['geometry']['lng']
        return lat, lon
    else:
        return None, None


def preprocess_image(image):
    img = np.array(image.resize((150, 150)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


st.title('Swachhta Monitoring and LiFE Practices')


uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    preprocessed_image = preprocess_image(image)


    timestamp = datetime.datetime.now()
    lat, lon = get_geolocation()


    prediction = model.predict(preprocessed_image)


    threshold = 0.7
    cleanliness_status = 1 if prediction >= threshold else 0


    st.write(f"**Timestamp:** {timestamp}")
    if lat and lon:
        st.write(f"**Location:** Latitude: {lat}, Longitude: {lon}")
    else:
        st.write("**Location:** Unable to fetch geolocation.")
    
    st.write(f"**Cleanliness Status:** {'Clean' if cleanliness_status == 1 else 'Not Clean'}")


    img_cv2 = np.array(image.convert('RGB'))
    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.title(f"Status: {'Clean' if cleanliness_status == 1 else 'Not Clean'}\n"
              f"Timestamp: {timestamp}\n"
              f"Location: {lat:.2f}, {lon:.2f}" if lat and lon else "Location Unavailable")
    st.pyplot(plt)
else:
    st.write("Please upload an image to analyze.")
