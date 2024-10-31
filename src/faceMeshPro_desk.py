import cv2
import numpy as np
import concurrent.futures
import time
import pyautogui
import os

def detect_faces(frame, net, min_confidence, save_folder):
    """
    Detect faces in the frame and save the cropped face images.

    Args:
        frame (numpy.ndarray): The image frame to process.
        net (cv2.dnn.Net): The pre-trained face detection model.
        min_confidence (float): Minimum confidence threshold.
        save_folder (str): Folder to save detected face images.
    """
    (h, w) = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // 4, h // 4))
    blob = cv2.dnn.blobFromImage(
        resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            cv2.rectangle(
                frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2
            )

            face_img = frame[start_y:end_y, start_x:end_x]
            face_img = cv2.resize(face_img, (300, 300))

            timestamp = str(time.time()).replace('.', '')
            img_name = os.path.join(save_folder, f"face_{timestamp}.jpg")

            cv2.imwrite(img_name, face_img)

def main():
    model_dir = './PhotoFaceDetect'
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(model_dir, 'deploy.prototxt'),
        os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    )

    min_confidence = 0.4
    save_folder = './face_photo/assets'
    os.makedirs(save_folder, exist_ok=True)
    capture_resolution = (1280, 720)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            screenshot = pyautogui.screenshot(
                region=(0, 0, capture_resolution[0], capture_resolution[1])
            )
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            future = executor.submit(
                detect_faces, frame, net, min_confidence, save_folder
            )

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            future.result()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
