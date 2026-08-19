"""
Pre-trained YOLO model download karo
Ye model pehle se trained hai, tumhe training ki zarurat nahi!
"""

import os
import urllib.request
from ultralytics import YOLO

def download_pretrained_model():
    """
    YOLOv8 base model download karo
    """
    print("🚀 Downloading YOLOv8 Model...")
    print("📦 Size: ~6MB, 1-2 minutes lagega")
    
    # Models folder banao
    os.makedirs('models', exist_ok=True)
    
    try:
        # YOLOv8n (nano) model download karo - fastest and smallest
        print("⏬ Downloading YOLOv8n (Nano) model...")
        model = YOLO('yolov8n.pt')
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Location: Current directory")
        print("\n💡 Note: Ye general object detection model hai")
        print("   License plate specific nahi hai, but kaam karega!")
        
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Model manually download karo:")
        print("   1. Visit: https://github.com/ultralytics/assets/releases")
        print("   2. Download: yolov8n.pt")
        print("   3. Save in project root folder")
        return None

if __name__ == "__main__":
    print("="*50)
    print("📥 YOLO MODEL DOWNLOADER")
    print("="*50)
    
    model = download_pretrained_model()
    
    if model:
        print("\n🎉 Setup Complete!")
        print("✅ Ab tum advanced_detector.py run kar sakte ho")
    
    print("="*50)