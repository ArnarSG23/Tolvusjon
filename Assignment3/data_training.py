import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='/Users/arnarsveinngudmundsson/Documents/HR/MSc/Onn_2/Tölvusjón/Assignments/yolov5/best.pt', source='local')

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]  # Annotated frame

    # Display the frame
    cv2.imshow('YOLOv5 Live Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()