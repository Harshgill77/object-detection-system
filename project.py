import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
# Use CPU backend (works on all systems)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


layer_name = net.getLayerNames()
output_layers = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]

cap=cv2.VideoCapture(0) #link here

# Optimized settings for speed
frame_skip = 3  # Process every 3rd frame instead of 2nd
input_size = (320, 320)  # Smaller input size for faster processing
confidence_threshold = 0.6  # Higher threshold to reduce false positives
nms_threshold = 0.3  # Lower NMS threshold for faster processing

frame_id = 0
fps_counter = 0
fps_start_time = time.time()
fps = 0  # Initialize fps variable

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    # Resize to smaller size for faster processing
    small_frame = cv2.resize(frame, input_size)
    blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, input_size, (0, 0, 0), swapRB=True)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detected in output:
            scores = detected[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:  # Higher threshold
                x, y, w, h = [int(detected[0] * frame.shape[1]), int(detected[1] * frame.shape[0]), 
                             int(detected[2] * frame.shape[1]), int(detected[3] * frame.shape[0])]
                boxes.append([x-w//2, y-h//2, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = tuple(map(int, colors[class_ids[i]]))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate and display FPS
    fps_counter += 1
    if fps_counter % 30 == 0:  # Update FPS every 30 frames
        current_time = time.time()
        fps = 30 / (current_time - fps_start_time)
        fps_start_time = current_time
    
    # Display FPS on screen
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame Skip: {frame_skip}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Optimized", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()