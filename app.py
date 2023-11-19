from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)

# Load the pre-trained model (make sure to provide the correct path and model name)
model = tf.keras.models.load_model('model_edged.h5')

# Define a mapping from class indices to ASL letters
class_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def preprocess_image(image):
    image = image.resize((100, 100))  # Resize to 100x100 pixels
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)  # Add a singleton dimension for the channel
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = image / 255.0  # Normalize to the range [0, 1]
    return image

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the entire frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply filters to the entire frame
        blured = cv2.GaussianBlur(gray, (5, 5), 0)
        blured = cv2.erode(blured, None, iterations=2)
        blured = cv2.dilate(blured, None, iterations=2)
        edged = cv2.Canny(blured, 50, 50)

        # Resize the processed image to the input size expected by the model
        model_input = cv2.resize(edged, (100, 100), interpolation=cv2.INTER_CUBIC)
        model_input = model_input.astype('float32') / 255.0
        model_input = model_input.reshape(1, 100, 100, 1)

        # Make a prediction
        prediction = model.predict(model_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        asl_letter = class_to_letter.get(predicted_class, 'Unknown')

        # Add prediction text to the original frame
        cv2.putText(frame, asl_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video-feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)