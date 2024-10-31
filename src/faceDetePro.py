import os
import time
import cv2
import numpy as np

def detect_faces(frame: np.ndarray, net: cv2.dnn_Net, min_confidence: float) -> None:
    """
    Detect faces in a frame and save them as images.

    Parameters:
        frame (np.ndarray): The image frame from the video stream.
        net (cv2.dnn_Net): The pre-loaded face detection model.
        min_confidence (float): Minimum confidence threshold for detections.
    """
    (h, w) = frame.shape[:2]
    # Prepare the image blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0,
                                 (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    os.makedirs('face_photo', exist_ok=True)

    # Iterate over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype(int)
            # Draw bounding box
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # Extract face ROI and save image
            face_img = frame[start_y:end_y, start_x:end_x]
            face_img = cv2.resize(face_img, (300, 300))
            img_name = f"face_photo/face_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(img_name, face_img)

def main():
    """
    Main function to perform face detection on webcam video stream.
    """
    # Load face detection model
    net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt',
                                   'models/res10_300x300_ssd_iter_140000.caffemodel')
    min_confidence = 0.8
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from webcam.")
                break
            # Detect faces in the frame
            detect_faces(frame, net, min_confidence)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
