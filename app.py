# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model (make sure to provide the correct path and model name)
model = load_model('final_asl_alphabet_model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame part in the request'}), 400

    file = request.files['frame']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Read the image and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((200, 200))
        image = np.array(image)
        if image.shape[-1] == 4:  # Check if image has an alpha channel
            image = image[..., :3]  # Convert from RGBA to RGB
        image = np.expand_dims(image, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        asl_letter = chr(predicted_class + 65)  # Assuming ASCII

        return jsonify({'prediction': asl_letter})

if __name__ == '__main__':
    app.run(debug=True)
