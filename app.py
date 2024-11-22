import cv2
import streamlit as st
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np

# Step 1: Load YOLO model for bounding box detection
yolo_model = YOLO("yolov5s.pt")  # Use a pre-trained YOLOv5 model

# Step 2: Load Vision Transformer for classification
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_model.eval()

# Step 3: Preprocessing function for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

def preprocess_vit(image):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

def classify_number_plate(image):
    with torch.no_grad():
        inputs = preprocess_vit(image)
        outputs = vit_model(inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax(dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    return predicted_class, confidence

def detect_and_classify(frame):
    # Detect bounding boxes using YOLO
    results = yolo_model(frame)
    detections = results.xyxy[0]  # Bounding box predictions

    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        if conf > 0.5:  # Filter by confidence
            label = "Number Plate"
            plate_region = frame[y1:y2, x1:x2]

            # Classify the plate content using ViT
            try:
                predicted_class, confidence = classify_number_plate(plate_region)
                label += f" ({predicted_class}, {confidence:.2f})"
            except Exception as e:
                label = "Error in classification"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    return frame

# Streamlit App
st.title("Real-Time Number Plate Detection and Classification")

# Camera input
run_camera = st.checkbox("Turn on Camera")

if run_camera:
    st.write("Camera is running...")

    # OpenCV to access webcam
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])  # Placeholder for the video feed

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Error accessing the camera.")
            break

        # Process the frame
        frame = detect_and_classify(frame)

        # Display in Streamlit
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Camera is off. Check the box above to enable it.")
