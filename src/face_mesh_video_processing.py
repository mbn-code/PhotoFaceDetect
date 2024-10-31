import cv2
import dlib
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import time
import concurrent.futures

# Define the face detection function
def detect_faces(frame, detector, min_confidence, save_folder, last_saved_face):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Initialize the list of faces
    face_images = []

    # Loop over the detected faces
    for face in faces:
        confidence = face.confidence

        # Filter out weak detections
        if confidence > min_confidence:
            # Extract the face region
            (x, y, w, h) = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
            face_img = frame[y:y+h, x:x+w]

            # Calculate similarity with the last saved face
            if last_saved_face is not None:
                similarity = calculate_similarity(face_img, last_saved_face)

                # If the similarity is below a threshold, save the new face image
                if similarity < 0.9:
                    face_images.append(face_img)

            # Update the last saved face
            last_saved_face = face_img

    # Save the face images
    if len(face_images) > 0:
        save_face_images(face_images, save_folder)

    # If no face is detected, return 0 confidence
    return 0, last_saved_face

# Function to calculate image similarity using Structural Similarity Index (SSIM)
def calculate_similarity(image1, image2):
    # If either image is smaller than 7x7, return similarity of 0
    if min(image1.shape) < 7 or min(image2.shape) < 7:
        return 0

    # Determine the appropriate window size based on the smaller image dimension
    win_size = min(7, min(image1.shape[:2]))

    similarity = ssim(image1, image2, win_size=win_size, multichannel=True)
    return similarity

# Function to save face images
def save_face_images(face_images, save_folder):
    # Generate unique filenames based on timestamps
    timestamps = [str(time.time()).replace('.', '') for _ in range(len(face_images))]
    img_names = [f"{save_folder}/face_{timestamp}.jpg" for timestamp in timestamps]

    # Save the face images
    for img_name, face_img in zip(img_names, face_images):
        cv2.imwrite(img_name, face_img)

# Load the face detection model
detector = dlib.get_frontal_face_detector()

# Set the minimum confidence threshold for face detection
min_confidence = 0.2  # Adjust this value as needed

# Specify the folder to save the face images
save_folder = r'C:\Users\manor\OneDrive\Dokumenter\Programming_folder\python\PhotoFaceDetect\face_photo'

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Set the video file path
video_path = 'C:/Users/manor/OneDrive/Dokumenter/Programming_folder/python/PhotoFaceDetect/test.mp4'

# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture(video_path)

# Initialize the last saved face
last_saved_face = None

# Read frames from the video file until the end
while True:
    # Read multiple frames to process in parallel
    frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if not frames:
        break

    # Process frames in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(detect_faces, frame, detector, min_confidence, save_folder, last_saved_face) for frame in frames]

    # Get the last saved face from the results
    _, last_saved_face = results[-1].result()

    # Display the frames
    for frame in frames:
        cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
