import cv2
import numpy as np
import concurrent.futures
import time
import pyautogui
import os

# Define the face detection function
def detect_faces(frame, net, min_confidence, save_folder):
    (h, w) = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // 4, h // 4))  # Resize the frame to half its resolution
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Zoom in on the detected face
            face_img = frame[startY:endY, startX:endX]
            face_img = cv2.resize(face_img, (300, 300))

            # Generate a unique filename based on timestamp
            timestamp = str(time.time()).replace('.', '')
            img_name = f"{save_folder}/face_{timestamp}.jpg"

            # Save the face image
            cv2.imwrite(img_name, face_img)

    # If no face is detected, return 0 confidence
    return 0

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('C:/Users/manor/OneDrive/Dokumenter/Programming_folder/python/PhotoFaceDetect/deploy.prototxt', 'C:/Users/manor/OneDrive/Dokumenter/Programming_folder/python/PhotoFaceDetect/res10_300x300_ssd_iter_140000.caffemodel')

# Set the minimum confidence threshold for face detection
min_confidence = 0.4

# Specify the folder to save the face images
save_folder = r'C:\Users\manor\OneDrive\Dokumenter\Programming_folder\python\PhotoFaceDetect\face_photo'

# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Set the capture resolution to 1920x1080
capture_resolution = (1280, 720)

# Create a thread pool
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Loop over capturing desktop frames
while True:
    # Capture screenshot of the desktop with the desired resolution
    screenshot = pyautogui.screenshot(region=(0, 0, capture_resolution[0], capture_resolution[1]))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Submit the frame processing to the thread pool
    future = executor.submit(detect_faces, frame, net, min_confidence, save_folder)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

    # Wait for the frame processing to complete
    future.result()

# Close all windows
cv2.destroyAllWindows()
