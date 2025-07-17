from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, Response
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import os
import cv2
from voice_alert import *

app = Flask(__name__)
app.secret_key = 'xyzsdfg'

# Load the trained model
MODEL_PATH = 'tail_rope_fault_detection.h5'  # Path to your .h5 model file
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ['Faulty', 'Healthy']

# Function to preprocess the uploaded image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert the image to RGB
    img = ImageOps.fit(image, (299, 299), Image.LANCZOS)  # Resize the image to 299x299
    img = np.array(img) / 255.0  # Normalize the image to [0, 1] range
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 299, 299, 3)
    return img

# Function to predict the fault type (Healthy or Faulty)
def predict_fault(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return CLASS_NAMES[int(prediction[0] > 0.5)]  # Binary classification threshold 0.5

# Route to capture video from the webcam and make predictions
@app.route('/camera')
def camera():
    return render_template('camera.html')

def generate_frames():
    from voice_alert import thread_safe_speak
    cap = cv2.VideoCapture(0)  # Open the default webcam
    fault_announced = False  # Flag to track if we've already announced a fault
    count=0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Predict the fault from the current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        label = predict_fault(image)

        # Draw prediction text on the frame
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If a fault is detected, trigger the alarm
        if label == "Faulty":
            cv2.putText(frame, "⚠️ Faulty rope detected!...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not fault_announced:
                fault_announced = True
                thread_safe_speak()

        # Encode the frame as a JPEG

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to be displayed on the webpage
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap.isOpened():
        cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Default route redirects to login
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form:
        email = request.form['email']
        # Store email in session
        session['loggedin'] = True
        session['email'] = email
        session['name'] = email.split('@')[0]  # Use part of email as name
        return redirect(url_for('user'))
    return render_template('login.html', message=message)

@app.route('/user')
def user():
    # Check if user is logged in
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    return render_template('user.html')

@app.route('/logout')
def logout():
    # Clear session data
    session.pop('loggedin', None)
    session.pop('email', None)
    session.pop('name', None)
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/photo', methods=['GET', 'POST'])
def photo():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded image temporarily in the uploads folder
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Load the image using PIL and predict
        image = Image.open(file)  # Open the uploaded file
        label = predict_fault(image)  # Pass the PIL image to predict_fault

        return render_template('result.html', label=label, filename=file.filename)

    return render_template('upload.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded video temporarily in the uploads folder
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)

        # Process the video for final prediction
        cap = cv2.VideoCapture(video_path)
        final_predictions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the captured frame to RGB for prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            label = predict_fault(image)
            final_predictions.append(label)

        cap.release()

        # Show final prediction
        final_prediction = final_predictions[-1] if final_predictions else "Unknown"
        return render_template('result1.html', label=final_prediction, filename=file.filename, predictions=final_predictions)

    return render_template('upload_video.html', video=True)

@app.route('/back')
def back():
    return render_template('user.html')

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)