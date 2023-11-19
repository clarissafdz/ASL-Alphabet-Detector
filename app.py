from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
model = tf.keras.models.load_model('asl_alphabet_model.h5')  # Ensure the model path is correct


@app.route('/')
def index():
    # Render your HTML page
    return app.send_static_file('index.html')

@app.route('/predict-asl', methods=['POST'])
def predict_asl():
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image and preprocess it for the model
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((200, 200))
        image = np.expand_dims(image, axis=0)
        image = np.array(image) / 255.0  # Normalize the image
        
        # Predict the class of the image
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming class indices are labels
        
        # Convert the class index into the corresponding ASL letter
        asl_letter = chr(predicted_class + 65)  # 65 is the ASCII for 'A'

        return jsonify({'prediction': asl_letter})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Define the PORT on which you want to run the server
    PORT = os.environ.get('PORT', 5000)
    # Start the Flask server
    app.run(host='0.0.0.0', port=PORT, debug=True)
