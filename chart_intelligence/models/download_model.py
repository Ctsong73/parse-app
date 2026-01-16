"""
Script to download a YOLO model for testing
"""
from ultralytics import YOLO
import os

def download_test_model():
    """Download a small YOLOv8 model for testing"""
    print("Downloading YOLOv8n model...")
    
    # Download nano model (smallest)
    model = YOLO('yolov8n.pt')
    
    # Save it as our chart model
    model.save('yolo_chart.pt')
    
    print(f"Model saved to: {os.path.abspath('yolo_chart.pt')}")
    print("Model size:", os.path.getsize('yolo_chart.pt'), "bytes")

if __name__ == "__main__":
    download_test_model()
