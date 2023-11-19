// Assumes you have a video element and a canvas element in your HTML
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('captureButton');
const predictedLetter = document.getElementById('predictedLetter');

// Set up the video stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error("Error accessing the video stream", error);
    });

// Capture the current video frame, draw it on the canvas, and send it to the server
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to an image format
    canvas.toBlob(blob => {
        // Create a FormData object to send the captured image
        const formData = new FormData();
        formData.append('file', blob, 'gesture.png');

        // Send the image to the Flask server
        fetch('/predict-asl', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display the predicted letter
            predictedLetter.textContent = data.prediction;
        })
        .catch(error => {
            console.error('Error predicting the ASL alphabet symbol:', error);
        });
    }, 'image/png');
});
