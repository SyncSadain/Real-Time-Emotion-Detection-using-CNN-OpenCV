import cv2
from keras.models import Sequential, model_from_json
from keras.saving import register_keras_serializable
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for

# Registering the custom Sequential class
@register_keras_serializable()
class MySequential(Sequential):
    pass

# Load the model architecture from JSON
with open("models/emotiondetector1.json", "r") as json_file:
    model_json = json_file.read()

# Load the model architecture and weights
model = model_from_json(model_json, custom_objects={"Sequential": Sequential, "MySequential": MySequential})
model.load_weights("models/emotiondetector1.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define the labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for flashing messages

# Video streaming generator function
def gen():
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            confidence = pred.max() * 100
            
            # Display sentiment and confidence
            text = f'Sentiment: {prediction_label}'
            confidence_text = f'Confidence: {confidence:.1f}%'
            name_text = 'Emotion Detection made by SADAIN'

            # Put text on the frame
            cv2.putText(frame, name_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to the login page and handle login logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('livestream'))
    return render_template('login.html')

# Route to the live stream page
@app.route('/livestream')
def livestream():
    return render_template('livestream.html')

@app.route('/register')
def register():
    return render_template('register.html')

# Route to the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
