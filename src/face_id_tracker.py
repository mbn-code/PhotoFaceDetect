import cv2
import numpy as np
import time
import os


class PersonTracker:
    """Class for tracking person IDs."""

    def __init__(self):
        self._face_id = 0
        self._person_ids = {}
        self._saved_ids = set()

    def get_next_face_id(self):
        """Increment and return the next face ID."""
        self._face_id += 1
        return self._face_id

    def update_person_id(self, face_id, person_id):
        """Update mapping from face ID to person ID."""
        self._person_ids[face_id] = person_id

    def get_person_id(self, face_id):
        """Get the person ID corresponding to a face ID."""
        return self._person_ids.get(face_id)

    def is_person_id_saved(self, person_id):
        """Check if a person ID has been saved."""
        return person_id in self._saved_ids

    def mark_person_id_as_saved(self, person_id):
        """Mark a person ID as saved."""
        self._saved_ids.add(person_id)

    def get_next_unique_person_id(self):
        """Get the next unique person ID that hasn't been saved."""
        while True:
            new_person_id = self.get_next_face_id()
            if not self.is_person_id_saved(new_person_id):
                return new_person_id


def detect_skin_color(directory='detected_faces'):
    """Detect average skin color from images in a directory."""
    avg_colors = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate the average RGB values
            avg_color = image_rgb.mean(axis=(0, 1))
            avg_r, avg_g, avg_b = avg_color

            avg_colors[filename] = (avg_r, avg_g, avg_b)

    return avg_colors


def calculate_brightness(rgb):
    """Calculate brightness of an RGB color using luminance formula."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def apply_studio_light(frame, start_x, start_y, end_x, end_y, brightness=1.5):
    """
    Apply Studio Light effect by adjusting the brightness of the face region.
    """
    # Extract the face region from the frame
    face = frame[start_y:end_y, start_x:end_x]

    # Apply the Studio Light effect
    adjusted_face = cv2.convertScaleAbs(face, alpha=brightness, beta=0)
    frame[start_y:end_y, start_x:end_x] = adjusted_face


def save_detected_face(frame, start_x, start_y, end_x, end_y, person_id, person_tracker):
    """Save the detected face as an image and mark the person ID as saved."""
    # Create the "detected_faces" directory if it doesn't exist
    os.makedirs("detected_faces", exist_ok=True)

    # Generate a unique filename for the saved face image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detected_faces/face_{person_id}_{timestamp}.jpg"

    # Save the detected face as an image
    face_image = frame[start_y:end_y, start_x:end_x]
    cv2.imwrite(filename, face_image)

    # Mark the person ID as saved
    person_tracker.mark_person_id_as_saved(person_id)


def detect_faces(frame, net, min_confidence, person_tracker):
    """
    Detect faces in a frame using a pre-trained DNN model.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Initialize the highest confidence as 0
    max_confidence = 0

    # Determine the overall brightness of the frame
    frame_brightness = frame.mean()

    # Check if the frame is in low light condition
    is_low_light = frame_brightness < 100  # Adjust threshold as needed

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )
            start_x, start_y, end_x, end_y = box.astype(int)

            # Draw the bounding box around the face
            cv2.rectangle(
                frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2
            )

            # Get the face ID
            face_id = i

            # Check if the face ID already has a person ID assigned
            person_id = person_tracker.get_person_id(face_id)
            if person_id is None:
                # Assign a new unique person ID
                person_id = person_tracker.get_next_unique_person_id()
                person_tracker.update_person_id(face_id, person_id)

            # Display the person ID
            text = f"Person ID: {person_id}"
            cv2.putText(
                frame,
                text,
                (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Update the maximum confidence if needed
            if confidence > max_confidence:
                max_confidence = confidence

            # Apply Studio Light effect to the face region only in low light
            if is_low_light:
                apply_studio_light(frame, start_x, start_y, end_x, end_y)

            # Save the detected face only if it hasn't been saved before
            if not person_tracker.is_person_id_saved(person_id):
                save_detected_face(
                    frame, start_x, start_y, end_x, end_y, person_id, person_tracker
                )

    # Print whether it's a low light condition or not
    brightness_text = "Low Light" if is_low_light else "Normal Light"
    cv2.putText(
        frame,
        brightness_text,
        (18, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        4,
    )
    return max_confidence


def process_frame(frame, net, min_confidence, person_tracker, start_time):
    """
    Process a video frame by detecting faces and calculating the FPS.
    """
    # Perform face detection on the frame and get the confidence score
    confidence = detect_faces(frame, net, min_confidence, person_tracker)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)  # Avoid division by zero

    # Set the color of the FPS text based on the FPS value
    if fps < 20:
        fps_color = (0, 0, 255)  # Red
    elif 20 <= fps <= 50:
        fps_color = (0, 165, 255)  # Orange
    else:
        fps_color = (0, 255, 0)  # Green

    # Draw FPS and confidence text on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        frame,
        fps_text,
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        fps_color,
        4,
    )

    confidence_text = f"Confidence: {confidence:.2f}"
    cv2.putText(
        frame,
        confidence_text,
        (18, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        4,
    )


def main():
    """Main function to run the face detection."""
    # Load the face detection model
    try:
        net = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel'
        )
    except cv2.error as e:
        print(f"Error loading model: {e}")
        return

    # Set the minimum confidence threshold for face detection
    min_confidence = 0.35

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Create a person tracker object
    person_tracker = PersonTracker()

    try:
        # Loop over the frames from the video stream
        while True:
            # Read the next frame from the video stream
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            start_time = time.time()

            # Process the frame
            process_frame(
                frame, net, min_confidence, person_tracker, start_time
            )

            # Display the frame
            cv2.imshow('Frame', frame)

            # Check if the user pressed the 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

    # Call the skin color detection function
    average_skin_colors = detect_skin_color('detected_faces')

    # Sort the results by brightness
    sorted_skin_colors = sorted(
        average_skin_colors.items(),
        key=lambda item: calculate_brightness(item[1]),
    )

    # Print the sorted results
    for filename, avg_color in sorted_skin_colors:
        print(f'Average skin color of {filename}: {avg_color}')


if __name__ == "__main__":
    main()