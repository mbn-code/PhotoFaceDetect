import cv2
import numpy as np
import os

def detect_faces(frame, net, min_confidence):
    """
    Detect faces in an image using a pre-trained DNN model.

    :param frame: The image frame in which to detect faces.
    :param net: The pre-trained DNN face detection model.
    :param min_confidence: The minimum confidence threshold to filter weak detections.
    """
    (h, w) = frame.shape[:2]
    # Preprocess the image to create a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_images = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            # Compute coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype(int)

            # Ensure coordinates are within the frame dimensions
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)

            # Draw the bounding box around the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                          (0, 0, 255), 2)

            # Extract the face ROI and resize it
            face_img = frame[start_y:end_y, start_x:end_x]
            if face_img.size == 0:
                continue  # Skip if the face ROI is empty
            face_img = cv2.resize(face_img, (300, 300))
            face_images.append((face_img, i))

    return face_images

def main():
    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',
                                   'models/res10_300x300_ssd_iter_140000.caffemodel')

    min_confidence = 0.3

    cap = cv2.VideoCapture(0)

    # Create directory for saving face images
    os.makedirs('face_photo', exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        face_images = detect_faces(frame, net, min_confidence)

        # Save the detected faces to files
        for face_img, idx in face_images:
            img_name = f"face_photo/face_{idx}.png"
            cv2.imwrite(img_name, face_img)

        # Display the frame with bounding boxes
        cv2.imshow('frame', frame)

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
