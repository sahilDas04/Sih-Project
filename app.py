import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import datetime
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

MODEL_PATH = 'swachhta_model_optimized_v2.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file)
        image_size = image.size
        preprocessed_image = preprocess_image(image)
        timestamp = datetime.datetime.now()
        prediction = model.predict(preprocessed_image)[0]

        if len(prediction) != len(class_labels):
            return jsonify({'error': f"Model prediction length {len(prediction)} does not match class labels length {len(class_labels)}"}), 500

        threshold = 0.7
        cleanliness_status = 'Clean' if max(prediction) >= threshold else 'Not Clean'
        percentages = (prediction / np.sum(prediction)) * 100
        summary = garbage_summary(percentages, class_labels)
        top_label = class_labels[np.argmax(prediction)]  
        confidence_score = max(prediction) 

        
        response = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'cleanliness_status': cleanliness_status,
            'predictions': {label: float(f"{percent * 100:.2f}") for label, percent in zip(class_labels, prediction)},  
            'summary': summary,
            'top_prediction': {
                'label': top_label,
                'confidence_score': float(confidence_score) * 100  
            },
            'raw_prediction': [float(p) * 100 for p in prediction],  
            'image_dimensions': {
                'width': image_size[0],
                'height': image_size[1]
            },
            'threshold': float(threshold) * 100 
        }


        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
