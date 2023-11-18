document.addEventListener("DOMContentLoaded", function () {
    const videoElement = document.getElementById('video');
    const exitButton = document.getElementById('exitButton');

    // Prompt user for permission to use the webcam and display the video feed
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                videoElement.srcObject = stream;
            })
            .catch(function (err) {
                console.log("Something went wrong!", err);
            });
    }

    // Stop the video stream when the exit button is clicked
    exitButton.addEventListener('click', function () {
        const stream = videoElement.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(function (track) {
                track.stop();
            });
        }
        videoElement.srcObject = null;
    });
});