#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}       Welcome to the Face Detection App       ${NC}"
    echo -e "${BLUE}========================================${NC}\n"

    echo -e "${YELLOW}Please select an option:${NC}"
    echo "1) Run face_mesh_desktop_gpu"
    echo "2) Run face_mesh"
    echo "3) Run face_id_tracker"
    echo "4) Run face_id_photo_capture"
    echo "5) Run advanced_face_detection"
    echo "6) Run enhanced_face_detection"
    echo "7) Run face_mesh_video_processing"
    echo "8) Run YOLOv3-tiny Face Detection"
    echo "9) Update Application"
    echo "10) Show Application Version"
    echo "11) View Logs"
    echo "0) Exit"

    read -p "Enter your choice: " choice

    case $choice in
        1)
            python3 src/face_mesh_desktop_gpu.py
            ;;
        2)
            python3 src/face_mesh.py
            ;;
        3)
            python3 src/face_id_tracker.py
            ;;
        4)
            python3 src/face_id_photo_capture.py
            ;;
        5)
            python3 src/advanced_face_detection.py
            ;;
        6)
            python3 src/enhanced_face_detection.py
            ;;
        7)
            python3 src/face_mesh_video_processing.py
            ;;
        8)
            python3 src/yolov3_tiny_face_detection.py
            ;;
        9)
            echo -e "\n${BLUE}Updating application...${NC}"
            git pull
            ;;
        10)
            echo -e "\n${GREEN}Face Detection App Version: 2.4.1${NC}"
            ;;
        11)
            echo -e "\n${BLUE}Displaying logs...${NC}"
            tail -n 20 logs/app.log
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