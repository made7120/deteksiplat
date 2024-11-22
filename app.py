import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import av
import numpy as np

# Load YOLO model for number plate detection
yolo_model = YOLO("yolov5s.pt")  # Change to "yolov5n.pt" for faster detection

# Load Vision Transformer (ViT) model for classification
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_model.eval()

# Preprocessing function for ViT
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

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detect bounding boxes using YOLO
        results = yolo_model(img)
        detections = results.xyxy[0]  # Bounding box predictions

        for detection in detections:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])
            if conf > 0.5:  # Filter by confidence
                label = "Number Plate"
                plate_region = img[y1:y2, x1:x2]

                # Classify the plate content using ViT
                try:
                    predicted_class, confidence = classify_number_plate(plate_region)
                    label += f" ({predicted_class}, {confidence:.2f})"
                except Exception as e:
                    label = "Error in classification"

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit application
st.title("Real-Time Number Plate Detection with Camera")

st.write("This application uses your device's camera to detect and classify vehicle number plates in real-time. Please allow camera access to continue.")

# WebRTC Camera Stream
webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": {"exact": "environment"}},  # Use back camera
        "audio": False,  # Disable audio
    },
)
