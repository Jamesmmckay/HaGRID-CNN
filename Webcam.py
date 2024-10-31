import threading
from PIL import Image
import cv2
import numpy as np
from AlexNet import AlexNet
import torch
import torchvision.transforms as transforms
import os
import utils
import json
import time

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

num_classes = 18
model = AlexNet(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

# Load and invert the label map
label_map = utils.get_label_map(config.get("file_path_2"))
label_map = {v: k for k, v in label_map.items()}

# Video capture setup
cap = cv2.VideoCapture(0)
frame_count = 0
predicted_label = None
lock = threading.Lock()

def predict_label():
    global frame_count, predicted_label
    while True:
        time.sleep(0.03)  # Run the prediction thread at a slower rate
        if frame_count % 30 == 0:
            with lock:
                if 'frame_for_prediction' in globals():
                    transformed_image = transform(frame_for_prediction)
                    transformed_image = transformed_image.unsqueeze(0)
                    
                    with torch.no_grad():  # Disable gradients for inference
                        output = model(transformed_image)
                        _, predicted = torch.max(output, 1)
                    
                    predicted_label = label_map.get(predicted.item(), "Unknown")
                    print(f"Predicted class: {predicted_label}")

# Start prediction thread
prediction_thread = threading.Thread(target=predict_label)
prediction_thread.daemon = True
prediction_thread.start()

# Main loop for capturing frames
while True:
    _, frame = cap.read()
    frame_count += 1

    with lock:
       frame_for_prediction = frame.copy()  # Save the current frame for the prediction thread

    # Display the frame and edge-detected frame
    edge = cv2.Canny(frame, 100, 100)

    cv2.imshow('Edges', edge)

    if predicted_label:
        cv2.putText(frame, f"Predicted: {predicted_label}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
