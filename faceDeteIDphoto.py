import cv2
import numpy as np

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

    # Initialize the highest confidence as 0
    max_confidence = 0
    total_confidence = 0
    num_faces = 0

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

            # Display the person ID
            text = f"ID: {person_id}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Update the maximum confidence if needed
            if confidence > max_confidence:
                max_confidence = confidence

            # Update the total confidence and number of faces
            total_confidence += confidence
            num_faces += 1

    # Calculate the average confidence
    if num_faces > 0:
        average_confidence = total_confidence / num_faces
    else:
        average_confidence = 0

    # If no face is detected, set the average confidence to 0
    if num_faces == 0:
        average_confidence = 0

    return max_confidence, average_confidence

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Set the minimum confidence threshold for face detection
min_confidence = 0.3

# Specify the path to the photo
photo_path = 'faces.jpg'

# Read the photo
frame = cv2.imread(photo_path)

# Create a person tracker object
person_tracker = PersonTracker()

# Process the photo and get the max confidence and average confidence
max_confidence, average_confidence = detect_faces(frame, net, min_confidence, person_tracker)

# Display the photo with the detected faces
cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
cv2.putText(frame, f"Max Confidence: {max_confidence:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.rectangle(frame, (10, 60), (300, 110), (0, 0, 0), -1)
cv2.putText(frame, f"Average Confidence: {average_confidence:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imshow('frame', frame)
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
