import streamlit as st
import numpy as np
import datetime
import random
import folium
from opencage.geocoder import OpenCageGeocode
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# API key for OpenCage Geocoding
API_KEY = '7c2e1bab2c41498191953e3110fdee4a'
geocoder = OpenCageGeocode(API_KEY)

# Load your trained model
MODEL_PATH = 'garbage_classification_model_old.h5'
try:
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Assuming these are your garbage types
class_labels = ['Plastic', 'Organic', 'Paper', 'Metal', 'Glass']

# Function to get random geolocation for demonstration
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

# Preprocessing the image before feeding to the model
def preprocess_image(image):
    img = np.array(image.resize((150, 150)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app title
st.title('Swachhta Monitoring and LiFE Practices')

# Image upload option
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)

    # Get current timestamp and geolocation
    timestamp = datetime.datetime.now()
    lat, lon = get_geolocation()

    # Predict using the model
    try:
        prediction = model.predict(preprocessed_image)
        # For multi-class classification
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_class]
        
        # Cleanliness threshold logic
        threshold = 0.7
        cleanliness_status = 1 if prediction[0][predicted_class] >= threshold else 0

        # Display results
        st.write(f"**Timestamp:** {timestamp}")
        if lat and lon:
            st.write(f"**Location:** Latitude: {lat}, Longitude: {lon}")

            # Create a folium map centered on the location
            m = folium.Map(location=[lat, lon], zoom_start=12)
            folium.Marker([lat, lon], popup=f"Location: {lat}, {lon}").add_to(m)

            # Save map to a BytesIO object
            map_data = BytesIO()
            m.save(map_data, close_file=False)
            st.subheader("Location Map")
            st.components.v1.html(map_data.getvalue().decode(), height=500, width=700)
        else:
            st.write("**Location:** Unable to fetch geolocation.")
        
        st.write(f"**Cleanliness Status:** {'Clean' if cleanliness_status == 1 else 'Not Clean'}")
        st.write(f"**Garbage Type:** {predicted_label}")

        # Display image with matplotlib and status overlay
        fig, ax = plt.subplots()
        ax.imshow(np.array(image.convert('RGB')))
        ax.set_title(f"Status: {'Clean' if cleanliness_status == 1 else 'Not Clean'}\n"
                     f"Garbage Type: {predicted_label}\n"
                     f"Timestamp: {timestamp}\n"
                     f"Location: {lat:.2f}, {lon:.2f}" if lat and lon else "Location Unavailable")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.write("Please upload an image to analyze.")
