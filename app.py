import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = 'swachhta_model_optimized_v2.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Class labels for the model
class_labels = ['Plastic', 'Organic', 'Paper', 'Metal']

# Preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image.resize((150, 150)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Generate a summary of the prediction
def garbage_summary(percentages, labels):
    max_index = np.argmax(percentages)
    max_label = labels[max_index]
    max_value = percentages[max_index]

    excessive_threshold = 50  # You can adjust this threshold

    if max_value >= excessive_threshold:
        return f"There is excessive {max_label} trash ({max_value:.2f}%). Consider focusing on cleaning up the {max_label} waste."
    else:
        return "The garbage composition is relatively balanced, but all types should be cleaned up."

# Convert a plot to a base64 string to return as a response
def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# API route to accept image upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file
        image = Image.open(file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Get current timestamp
        timestamp = datetime.datetime.now()

        # Make the prediction
        prediction = model.predict(preprocessed_image)[0]  # Get prediction for the uploaded image

        # Check if the prediction and labels match in size
        if len(prediction) != len(class_labels):
            return jsonify({'error': f"Model prediction length {len(prediction)} does not match class labels length {len(class_labels)}"}), 500

        # Determine if it's "clean" or "not clean" based on threshold
        threshold = 0.7
        cleanliness_status = 'Clean' if max(prediction) >= threshold else 'Not Clean'

        # Compute percentages of each garbage type
        percentages = (prediction / np.sum(prediction)) * 100

        # Generate garbage summary
        summary = garbage_summary(percentages, class_labels)

        # Generate pie chart for the response
        fig, ax = plt.subplots()
        ax.pie(percentages, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        pie_chart_base64 = plot_to_base64(fig)  # Convert plot to base64 string

        # Prepare the response
        response = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'cleanliness_status': cleanliness_status,
            'predictions': {label: float(f"{percent*100:.2f}") for label, percent in zip(class_labels, prediction)},
            'summary': summary,
            'pie_chart': pie_chart_base64  # Return the chart as a base64 string
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
