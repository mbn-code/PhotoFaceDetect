import sys
import cv2
import numpy as np
from pathlib import Path
import logging


class PersonTracker:
    """Class to track person IDs across detected faces."""

    def __init__(self):
        self.face_id = 0
        self.person_ids = {}

    def get_person_id(self):
        """Assign a new person ID."""
        self.face_id += 1
        return self.face_id

    def update_person_id(self, face_id, person_id):
        """Update the mapping from face ID to person ID."""
        self.person_ids[face_id] = person_id

    def get_person_id_from_face_id(self, face_id):
        """Retrieve the person ID associated with a face ID."""
        return self.person_ids.get(face_id)


def detect_faces(frame, net, min_confidence, person_tracker):
    """Detect faces in a frame using a pre-trained neural network.

    Args:
        frame (np.ndarray): The image frame in which to detect faces.
        net (cv2.dnn_Net): The pre-trained neural network for face detection.
        min_confidence (float): Minimum confidence threshold for detections.
        person_tracker (PersonTracker): An instance to track person IDs.

    Returns:
        tuple: A tuple containing the maximum and average confidence of detections.
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    max_confidence = 0.0
    total_confidence = 0.0
    num_faces = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype(int)

            # Draw the bounding box around the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

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
            cv2.putText(
                frame,
                text,
                (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Update the maximum confidence if needed
            max_confidence = max(max_confidence, confidence)

            # Update the total confidence and number of faces
            total_confidence += confidence
            num_faces += 1

    # Calculate the average confidence
    average_confidence = total_confidence / num_faces if num_faces > 0 else 0.0

    return max_confidence, average_confidence


def main():
    """Main function to execute face detection on a photo."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load the face detection model
    model_dir = Path("models")
    prototxt_path = model_dir / "deploy.prototxt"
    model_weights_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not prototxt_path.exists() or not model_weights_path.exists():
        logging.error(
            "Model files not found. Please ensure the 'models' directory contains the required files."
        )
        sys.exit(1)

    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_weights_path))

    # Set the minimum confidence threshold for face detection
    min_confidence = 0.3

    # Specify the path to the photo
    photo_path = Path("assets/images/faces.jpg")  # Update this path to your image file

    if not photo_path.exists():
        logging.error(f"Image file '{photo_path}' not found. Please check the file path.")
        sys.exit(1)

    # Read the photo
    frame = cv2.imread(str(photo_path))

    # Check if the image was loaded successfully
    if frame is None:
        logging.error(f"Could not read image '{photo_path}'. Please check the file.")
        sys.exit(1)

    # Create a person tracker object
    person_tracker = PersonTracker()

    # Process the photo and get the max confidence and average confidence
    max_confidence, average_confidence = detect_faces(
        frame, net, min_confidence, person_tracker
    )

    # Display the photo with the detected faces
    cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Max Confidence: {max_confidence:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.rectangle(frame, (10, 60), (300, 110), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Average Confidence: {average_confidence:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Face Detection", frame)
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
