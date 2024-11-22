import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
from ultralytics import YOLO
import av

# Load YOLO model
yolo_model = YOLO("yolov5n.pt")  # Lightweight YOLO model for faster detection

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detect bounding boxes using YOLO
        results = yolo_model(img)
        detections = results.xyxy[0]  # Get bounding box predictions

        for detection in detections:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])
            if conf > 0.5:  # Filter detections by confidence threshold
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Confidence: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit App
st.title("Real-Time Object Detection with YOLO and Streamlit")
st.write("This app detects objects in real-time using your device's camera. Please allow camera access to proceed.")

# Camera resolution options
st.sidebar.title("Camera Settings")
resolution = st.sidebar.selectbox("Select Camera Resolution", ["640x480", "1280x720", "1920x1080"])
width, height = map(int, resolution.split("x"))

# Camera facing mode options
camera_mode = st.sidebar.radio("Select Camera", ["Default", "Front Camera", "Back Camera"])

# Set facing mode based on user selection
if camera_mode == "Front Camera":
    facing_mode = {"facingMode": {"exact": "user"}}
elif camera_mode == "Back Camera":
    facing_mode = {"facingMode": {"exact": "environment"}}
else:
    facing_mode = True  # Default camera

# Start WebRTC Stream
try:
    webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": width},
                "height": {"ideal": height},
                **(facing_mode if isinstance(facing_mode, dict) else {}),
            },
            "audio": False,  # Disable audio to reduce load
        },
    )
except Exception as e:
    st.error(f"Error accessing the camera: {str(e)}")
    st.warning("Ensure your device has a functional camera and proper permissions for this application.")

st.sidebar.info("Note: If the camera does not load, ensure your browser has access to the camera and try switching resolutions or camera modes.")
