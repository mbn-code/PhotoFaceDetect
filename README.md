# PhotoFaceDetect

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Issues](https://img.shields.io/github/issues/yourusername/PhotoFaceDetect)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/yourusername/PhotoFaceDetect)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Download Pre-trained Models](#download-pre-trained-models)
- [Usage](#usage)
  - [Run Script](#run-script)
  - [Available Scripts](#available-scripts)
- [Workspace Structure](#workspace-structure)
- [Models](#models)
- [License](#license)
- [Contributing](#contributing)

## Introduction

**PhotoFaceDetect** is a Python-based application designed for real-time face detection, tracking, and analysis. Leveraging powerful libraries like OpenCV and YOLOv3-tiny, this project offers multiple scripts to cater to various face detection and processing needs. Whether you're capturing faces from your desktop screen or streaming from a webcam, PhotoFaceDetect provides efficient and customizable solutions.

## Features

- **Real-Time Face Detection:** Utilize OpenCV's DNN module and YOLOv3-tiny for efficient face detection.
- **Multithreading Optimization:** Enhance performance using Python's [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html).
- **Customizable Scripts:** Multiple scripts available for different face detection approaches.
- **Model Integration:** Supports various pre-trained models for age and gender detection.
- **Automated Photo Capture:** Save detected faces as separate image files with ease.

## Architecture

![Architecture Diagram](assets/images/architecture.png)

*Figure 1: Overview of PhotoFaceDetect Architecture.*

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/PhotoFaceDetect.git
cd PhotoFaceDetect
```

## Install Dependencies

### Ensure all required Python packages are installed

```bash
pip install -r requirements.txt
```

## Download Pre-trained Models

The project relies on several pre-trained models located in the `models` directory. Ensure you have the following files:

- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`
- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`
- `coco.names`

You can download these models from their respective official sources:

- [YOLOv3-tiny](https://link_to_yolov3_tiny)
- [Caffe Models](https://link_to_caffe_models)
- [COCO Names](https://link_to_coco_names)

Place the downloaded files in the `models` directory.

## Usage

### Run Script

To start the application, use the provided `run.sh` script. This script allows you to select and execute different face detection and processing scripts.

```bash
./run.sh
```

## Available Scripts

The `run.sh` script provides the following options:

- **Run faceMeshPro_desk_GPU**
  - **Executes**: `src/faceMeshPro_desk_GPU.py`
  - **Description**: Captures frames from the desktop screen and performs face detection using GPU acceleration.

- **Run faceMesh**
  - **Executes**: `faceMesh.py`
  - **Description**: Utilizes YOLOv3-tiny for real-time face detection from a webcam stream.

- **Run faceDeteID**
  - **Executes**: `faceDeteID.py`
  - **Description**: Detects faces and performs ID tracking with real-time display.

- **Run faceDeteIDphoto**
  - **Executes**: `faceDeteIDphoto.py`
  - **Description**: Detects faces from a video stream and saves them as photos.

- **Run faceDetePro**
  - **Executes**: `faceDetePro.py`
  - **Description**: Advanced face detection with multiple processing layers.

- **Run faceDetePro2**
  - **Executes**: `faceDetePro2.py`
  - **Description**: Enhanced face detection with skin color analysis.

- **Run faceMesh_pro_video**
  - **Executes**: `faceMesh_pro_video.py`
  - **Description**: Processes video streams for professional-grade face detection.

- **Run YOLOv3-tiny Face Detection**
  - **Executes**: `src/yolov3_tiny_face_detection.py`
  - **Description**: Implements YOLOv3-tiny for efficient face detection.

- **Exit**
  - Exits the application.

Upon running, you'll be prompted to select an option:

```bash
========================================
       Welcome to the Face Detection App       
========================================

Please select an option:
1) Run faceMeshPro_desk_GPU
2) Run faceMesh
3) Run faceDeteID
4) Run faceDeteIDphoto
5) Run faceDetePro
6) Run faceDetePro2
7) Run faceMesh_pro_video
8) Run YOLOv3-tiny Face Detection
0) Exit
Enter your choice:
```

## Workspace Structure

## Project Directory Structure

PhotoFaceDetect/  
├── assets/  
│   └── images/  
├── coco/  
│   └── coco.names  
├── models/  
│   ├── age_net.caffemodel  
│   ├── deploy_age.prototxt  
│   ├── deploy_gender.prototxt  
│   ├── deploy.prototxt  
│   ├── gender_net.caffemodel  
│   ├── model_CNN_V2.h5  
│   ├── res10_300x300_ssd_iter_140000.caffemodel  
│   ├── yolov3-tiny.cfg  
│   └── yolov3-tiny.weights  
├── src/  
│   ├── faceDeteID.py  
│   ├── faceDeteIDphoto.py  
│   ├── faceDetePro.py  
│   ├── faceDetePro2.py  
│   ├── faceMesh_pro_desk_GPU.py  
│   ├── faceMesh_pro_video.py  
│   ├── faceMesh.py  
│   └── ...  
├── .gitignore  
├── README.md  
├── requirements.txt  
└── run.sh  

## Directory Descriptions

- `assets/`: Contains static assets like images used in the project.
- `coco/`: Holds the COCO class labels file (`coco.names`).
- `detected_faces/`: Directory where detected face images are stored.
- `face_photo/`: Stores saved face photos.
- `models/`: Contains pre-trained models and their configuration files.
- `src/`: Source code for various face detection and processing scripts.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Project documentation.
- `requirements.txt`: Lists Python dependencies.
- `run.sh`: Bash script to execute different scripts based on user input.

## Models

The models directory includes several pre-trained models essential for face detection and analysis:

### Caffe Models

- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

### YOLOv3-tiny

- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`

### Age and Gender Detection

- `deploy_age.prototxt`
- `deploy_gender.prototxt`
- `age_net.caffemodel`
- `gender_net.caffemodel`
- `model_CNN_V2.h5`

Ensure all models are properly placed in the `models` directory for the scripts to function correctly.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Repository
2. Create a Feature Branch

```bash
git checkout -b feature/YourFeature
```

1. Commit Your Changes

```bash
git commit -m "Your Message"
```

```bash
git push origin feature/YourFeature
```

1. Open PR
