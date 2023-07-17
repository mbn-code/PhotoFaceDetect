import cv2
import numpy as np
import threading
import time

# Define a class to track person IDs
class PersonTracker:
    def __init__(self):
        self.face_id = 0
        self.person_ids = {}

    def get_person_id(self):
        self.face_id += 1
        return self.face_id

    def update_person_id(self, face_id, person_id):
        self.person_ids[face_id] = person_id

    def get_person_id_from_face_id(self, face_id):
        return self.person_ids.get(face_id)

# Define the face detection function
def detect_faces(frame, net, min_confidence, person_tracker):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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

            # Get the face ID
            face_id = i

            # Check if the face ID already has a person ID assigned
            person_id = person_tracker.get_person_id_from_face_id(face_id)
            if person_id is None:
                # Assign a new person ID
                person_id = person_tracker.get_person_id()
                person_tracker.update_person_id(face_id, person_id)

                # Save the image corresponding to the person ID
                face_img = frame[startY:endY, startX:endX]
                img_name = f"face_photo/face_{person_id}.jpg"
                cv2.imwrite(img_name, face_img)

            # Display the person ID
            text = f"Person ID: {person_id}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # If no face is detected, return 0 confidence
    return 0

# Define the thread function
def process_frame(frame, net, min_confidence, person_tracker):
    # Perform face detection on the frame
    confidence = detect_faces(frame, net, min_confidence, person_tracker)

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Set the minimum confidence threshold for face detection
min_confidence = 0.5

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Create a person tracker object
person_tracker = PersonTracker()

# Loop over the frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Create a thread for processing the current frame
    thread = threading.Thread(target=process_frame, args=(frame, net, min_confidence, person_tracker))

    # Start the thread
    thread.start()

    # Wait for the thread to finish before processing the next frame
    thread.join()

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
