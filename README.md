# PhotoFaceDetect
This repo contains files for detecting faces zooming in on them and taking photos, using multi threading to optimise performance and the YOLOv3-tiny classifier. 

# Face Detection with OpenCV and PyAutoGUI

This repository contains Python scripts that demonstrate face detection using OpenCV and PyAutoGUI. There are two separate scripts provided, each showcasing a different face detection approach.

## Requirements

To run the scripts, you need to have the following dependencies installed:

- Python 3
- OpenCV
- NumPy
- Concurrent Futures (for the first script)
- PyAutoGUI (for the first script)

## Script 1: Face Detection using OpenCV and PyAutoGUI

The first script (`face_detection_opencv_pyautogui.py`) captures frames from the desktop screen and performs face detection on them using OpenCV's DNN module. Detected faces are then saved as separate image files.

### Usage

1. Install the required dependencies using the following command:
```shell
pip install opencv-python numpy concurrent.futures pyautogui
```

2. Update the file paths in the script for the face detection model and the save folder.

3. Run the script using the following command:
```shell
python face_detection_opencv_pyautogui.py
```

4. The script will capture frames from the desktop screen and display them in a window. Detected faces will be highlighted with bounding boxes, and images of the faces will be saved in the specified folder.

5. Press the 'q' key to quit the script.

## Script 2: YOLOv3-tiny Face Detection

The second script (`yolov3_tiny_face_detection.py`) uses YOLOv3-tiny, a lightweight version of the YOLO (You Only Look Once) object detection algorithm, to detect faces in real-time video stream from a webcam.

### Usage

1. Install the required dependencies using the following command:
```shell
pip install opencv-python numpy
```

2. Download the YOLOv3-tiny weights file (`yolov3-tiny.weights`) and the configuration file (`yolov3-tiny.cfg`) from the official Darknet website or another reliable source.

3. Download the COCO class labels file (`coco.names`) from the official Darknet website or another reliable source.

4. Update the file paths in the script for the YOLOv3-tiny weights, configuration, and class labels.

5. Run the script using the following command:

```shell
python yolov3_tiny_face_detection.py
```

6. The script will open a window showing the video stream from the webcam. Detected faces will be highlighted with bounding boxes and confidence scores.

7. Press the 'q' key to quit the script.

**Note:** Make sure your computer has a webcam connected for script 2.

## License

This project is licensed under the [MIT License](LICENSE).
