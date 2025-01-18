from flask import Flask, request, render_template
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect and change hair and eye color with more precision
def change_color_of_hair_and_eyes(img, color):
    # Convert color input (hex) to BGR
    hex_color = color.lstrip('#')
    b, g, r = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert image to grayscale (needed for face detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Define the region of interest (ROI) for the face
        roi_face = img[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_face)
        
        # Apply color to detected eyes
        for (ex, ey, ew, eh) in eyes:
            roi_face[ey:ey+eh, ex:ex+ew] = (b, g, r)  # Apply BGR color to eyes

        # Apply color to detected hair (for simplicity, we take top part of the face as hair)
        hair_region = img[y:y+int(h/3), x:x+w]  # Rough assumption: Top third is hair
        hair_region[:, :] = (b, g, r)  # Apply BGR color to hair

    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    color = request.form['color']

    # Open image using PIL and convert to OpenCV format (BGR)
    img = Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply color change to hair and eyes
    edited_img = change_color_of_hair_and_eyes(img, color)

    # Save the edited image to the static folder
    output_path = 'static/edited_image.png'
    cv2.imwrite(output_path, edited_img)

    return render_template('result.html', image_path='static/edited_image.png')

if __name__ == '__main__':
    app.run(debug=True)
