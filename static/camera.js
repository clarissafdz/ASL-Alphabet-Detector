 document.addEventListener("DOMContentLoaded", function () {
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const exitButton = document.getElementById('exitButton');
    
    let model;
    handpose.load().then(loadedModel => {
        model = loadedModel;
        console.log("Handpose model loaded.");
    });

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoElement.srcObject = stream;
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                processVideo();
            };
        })
        .catch(err => {
            console.log("Something went wrong!", err);
        });

    // Process each video frame and draw landmarks
    function processVideo() {
        if (model && videoElement.readyState === 4) { // Check if model is loaded and video is ready
            model.estimateHands(videoElement).then(predictions => {
                drawHand(predictions);
                requestAnimationFrame(processVideo);
            });
        } else {
            requestAnimationFrame(processVideo); // Try again on the next frame
        }
    }

    function drawHand(predictions) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        predictions.forEach(prediction => {
            for (let i = 0; i < prediction.landmarks.length; i++) {
                const x = prediction.landmarks[i][0];
                const y = prediction.landmarks[i][1];
                canvasCtx.beginPath();
                canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
                canvasCtx.fillStyle = 'lime';
                canvasCtx.fill();
            }
        });
    }

    exitButton.addEventListener('click', function () {
        const stream = videoElement.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
        }
        videoElement.srcObject = null;
    });
}); 