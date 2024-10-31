#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}       Welcome to the Face Detection App       ${NC}"
    echo -e "${BLUE}========================================${NC}\n"

    echo -e "${YELLOW}Please select an option:${NC}"
    echo "1) Run faceMeshPro_desk_GPU"
    echo "2) Run faceMesh"
    echo "3) Run faceDeteID"
    echo "4) Run faceDeteIDphoto"
    echo "5) Run faceDetePro"
    echo "6) Run faceDetePro2"
    echo "7) Run faceMesh_pro_video"
    echo "8) Run YOLOv3-tiny Face Detection"
    echo "0) Exit"

    read -p "Enter your choice: " choice

    case $choice in
        1)
            python3 src/faceMeshPro_desk_GPU.py
            ;;
        2)
            python3 src/faceMesh.py
            ;;
        3)
            python3 src/faceDeteID.py
            ;;
        4)
            python3 src/faceDeteIDphoto.py
            ;;
        5)
            python3 src/faceDetePro.py
            ;;
        6)
            python3 src/faceDetePro2.py
            ;;
        7)
            python3 src/faceMesh_pro_video.py
            ;;
        8)
            python3 src/yolov3_tiny_face_detection.py
            ;;
        0)
            echo -e "\n${GREEN}Exiting... Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid choice. Please try again.${NC}"
            read -n 1 -s -r -p "Press any key to continue..."
            ;;
    esac
done