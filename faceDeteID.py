import cv2
import numpy as np
import threading
import time
import os

# Define a class to track person IDs
class PersonTracker:
    def __init__(self):
        self.face_id = 0
        self.person_ids = {}
        self.saved_ids = set()

    def get_person_id(self):
        self.face_id += 1
        return self.face_id

    def update_person_id(self, face_id, person_id):
        self.person_ids[face_id] = person_id

    def get_person_id_from_face_id(self, face_id):
        return self.person_ids.get(face_id)

    def is_person_id_saved(self, person_id):
        return person_id in self.saved_ids

    def mark_person_id_as_saved(self, person_id):
        self.saved_ids.add(person_id)

    def get_next_unique_person_id(self):
        while True:
            new_person_id = self.get_person_id()
            if not self.is_person_id_saved(new_person_id):
                return new_person_id
            
def detect_skin_color(directory='detected_faces'):
    avg_colors = {}

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image_rgb.shape
            total_pixels = h * w

            # Sum up the RGB values
            total_r = np.sum(image_rgb[:, :, 0])
            total_g = np.sum(image_rgb[:, :, 1])
            total_b = np.sum(image_rgb[:, :, 2])

            # Calculate the average RGB values
            avg_r = total_r // total_pixels
            avg_g = total_g // total_pixels
            avg_b = total_b // total_pixels

            avg_colors[filename] = (avg_r, avg_g, avg_b)

    return avg_colors

def calculate_brightness(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

def apply_studio_light(frame, startX, startY, endX, endY):
    # Define the Studio Light effect parameters
    brightness = 1.5  # Adjust the value as needed to control brightness

    # Extract the face region from the frame
    face = frame[startY:endY, startX:endX]

    # Apply the Studio Light effect by adjusting the brightness of the face region
    frame[startY:endY, startX:endX] = cv2.convertScaleAbs(face, alpha=brightness, beta=0)

def save_detected_face(frame, startX, startY, endX, endY, person_id):
    # Create the "detected_faces" directory if it doesn't exist
    if not os.path.exists("detected_faces"):
        os.makedirs("detected_faces")

    # Generate a unique filename for the saved face image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detected_faces/face_{person_id}_{timestamp}.jpg"

    # Save the detected face as an image
    cv2.imwrite(filename, frame[startY:endY, startX:endX])

    # Mark the person ID as saved
    person_tracker.mark_person_id_as_saved(person_id)

# Define the face detection function
def detect_faces(frame, net, min_confidence, person_tracker):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Initialize the highest confidence as 0
    max_confidence = 0

    # Determine the overall brightness of the frame
    brightness = frame.mean()

    # Check if the frame is in low light condition
    is_low_light = brightness < 100  # You can adjust this threshold based on your preference

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
                # Assign a new unique person ID
                person_id = person_tracker.get_next_unique_person_id()
                person_tracker.update_person_id(face_id, person_id)

            # Display the person ID
            text = f"Person ID: {person_id}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update the maximum confidence if needed
            if confidence > max_confidence:
                max_confidence = confidence

            # Apply Studio Light effect to the face region only in low light
            if is_low_light:
                apply_studio_light(frame, startX, startY, endX, endY)

            # Save the detected face only if it hasn't been saved before
            if not person_tracker.is_person_id_saved(person_id):
                save_detected_face(frame, startX, startY, endX, endY, person_id)

    # Print whether it's a low light condition or not
    brightness_text = "Low Light" if is_low_light else "Normal Light"
    cv2.putText(frame, brightness_text, (18, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    # If no face is detected, return 0 confidence
    return max_confidence

# Define the thread function
def process_frame(frame, net, min_confidence, person_tracker, start_time):
    # Perform face detection on the frame and get the confidence score
    confidence = detect_faces(frame, net, min_confidence, person_tracker)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Set the color of the FPS text based on the FPS value
    if fps < 20:
        fps_color = (0, 0, 255)  # Red
    elif 20 <= fps <= 50:
        fps_color = (0, 165, 255)  # Orange
    else:
        fps_color = (0, 255, 0)  # Green

    # Draw FPS and confidence text on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 4)

    confidence_text = f"Confidence: {confidence:.2f}"
    cv2.putText(frame, confidence_text, (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

# Load the face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Set the minimum confidence threshold for face detection
min_confidence = 0.35

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
    start_time = time.time()
    thread = threading.Thread(target=process_frame, args=(frame, net, min_confidence, person_tracker, start_time))

    # Start the thread
    thread.start()

    # Wait for the thread to finish before processing the next frame
    thread.join()

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Call the skin color detection function
average_skin_colors = detect_skin_color('detected_faces')

# Sort the results by brightness
sorted_skin_colors = sorted(average_skin_colors.items(), key=lambda item: calculate_brightness(item[1]))

# Print the sorted results
for filename, avg_color in sorted_skin_colors:
    print(f'Average skin color of {filename}: {avg_color}')

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()