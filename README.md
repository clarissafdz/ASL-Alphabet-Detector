## Inspiration
Our inspiration for this project was our initial idea. First we thought it would be fun to implement a software that allows users to turn on their camera, and then play Rock Paper Scissors with a computer. This idea would use a hand movement detector and AI to determine what sign the user was holding. However, after thinking about the project we realized that similar technology could be used to detect sign language and we decided this would be a much more helpful and useful idea.
## What it does
Recognize ASL Alphabet Signs: The system can detect and interpret the 26 ASL alphabet signs made by users in front of a camera. This includes signs for letters from A to Z in American Sign Language.
Real-Time Interpretation: It operates in real-time, making it capable of interpreting ASL signs as they are signed by users. This instantaneous feedback allows for smooth and fluid communication.
Translation to Text: The system translates recognized ASL signs into text, however, during the given time frame, we were only able to accomplish 70% accuracy.
User-Friendly Interface: We have designed an intuitive web interface that makes it easy for users to interact with the system. Users simply need to sign the ASL alphabet letters in front of their webcam, and the system does the rest.
## How we built it
We built this project using a combination of technologies, including computer vision libraries, deep learning models, and web development tools. We trained a custom ASL sign language detection model using a labeled dataset of ASL signs. The web application was created using HTML, CSS, and JavaScript, with the Flask framework on the backend to handle real-time predictions. We integrated the computer vision model into the web app to process video frames from the user's webcam, interpret the signs, and display the results on the web interface.
## Challenges we ran into
During the development of our project, we faced several challenges. These included:
-Training a robust ASL sign detection model with limited data.
-Implementing real-time video processing and prediction within a web application.
-Ensuring the accuracy and responsiveness of the system for various sign gestures.
-Managing and displaying predictions in a user-friendly manner on the web interface.
## Accomplishments that we're proud of
-Successfully training a custom ASL sign detection model that can recognize a variety of signs.
-Creating a user-friendly web interface that allows individuals to interact with the system easily.
-Implementing real-time video processing for continuous sign detection.
## What we learned
Throughout the development of this project, we learned valuable skills in computer vision, deep learning, web development, and user interface design. We gained insights into the challenges faced by the Deaf and Hard of Hearing community and how technology can be used to bridge communication gaps.
## What's next for Empowering Communication: ASL Alphabet Detector
In the future, we plan to enhance and expand our project by:
-Improving the accuracy and robustness of the ASL sign detection model.
-Adding support for a wider range of ASL signs and gestures.
-Incorporating translation into different languages.
-Exploring options for mobile app development to increase accessibility.
-Collaborating with the ASL community to gather feedback and make continuous improvements to the system.
