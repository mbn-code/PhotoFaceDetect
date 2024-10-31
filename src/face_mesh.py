import cv2
import numpy as np

def load_yolo_model():
    net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
    return net, output_layers

def load_classes():
    with open("coco/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    person_class_index = classes.index("person")
    return classes, person_class_index

def process_frame(frame, net, output_layers, person_class_index):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == person_class_index and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=0.5, nms_threshold=0.4
    )
    return boxes, confidences, indexes

def draw_labels(frame, boxes, confidences, indexes, colors):
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            color = colors[i % len(colors)].tolist()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{confidences[i]:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

def main():
    net, output_layers = load_yolo_model()
    classes, person_class_index = load_classes()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, indexes = process_frame(
            frame, net, output_layers, person_class_index
        )
        draw_labels(frame, boxes, confidences, indexes, colors)

        cv2.imshow("YOLOv3-tiny Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
