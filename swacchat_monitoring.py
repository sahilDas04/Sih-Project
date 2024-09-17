import streamlit as st
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore

MODEL_PATH = 'swachhta_model_optimized_v2.h5'
try:
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

class_labels = ['Plastic', 'Organic', 'Paper', 'Metal']

def preprocess_image(image):
    img = np.array(image.resize((150, 150)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def garbage_summary(percentages, labels):
    max_index = np.argmax(percentages)
    max_label = labels[max_index]
    max_value = percentages[max_index]
    
    excessive_threshold = 50

    if max_value >= excessive_threshold:
        return f"There is excessive {max_label} trash ({max_value:.2f}%). Consider focusing on cleaning up the {max_label} waste."
    else:
        return "The garbage composition is relatively balanced, but all types should be cleaned up."

st.title('Swachhta Monitoring and LiFE Practices')

st.subheader("Upload an Image")
uploaded_image = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)

    timestamp = datetime.datetime.now()

    try:
        prediction = model.predict(preprocessed_image)[0]  

        if len(prediction) != len(class_labels):
            raise ValueError(f"Model prediction length {len(prediction)} does not match class labels length {len(class_labels)}")

        threshold = 0.7
        cleanliness_status = 1 if max(prediction) >= threshold else 0

        st.write(f"**Timestamp:** {timestamp}")

        st.write(f"**Cleanliness Status:** {'Clean' if cleanliness_status == 1 else 'Not Clean'}")

        st.subheader("Garbage Types and Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"**{label}:** {prediction[i] * 100:.2f}%")

        percentages = (prediction / np.sum(prediction)) * 100  

        summary = garbage_summary(percentages, class_labels)
        st.subheader("Garbage Composition Summary")
        st.write(summary)

        fig, ax = plt.subplots()
        ax.pie(percentages, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        st.subheader("Garbage Composition Percentage")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.imshow(np.array(image.convert('RGB')))
        ax.set_title(f"Status: {'Clean' if cleanliness_status == 1 else 'Not Clean'}\n"
                     f"Garbage Types: {', '.join(class_labels)}\n"
                     f"Timestamp: {timestamp}")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.write("Please upload an image to analyze.")
