"""
ADVANCED NUMBER PLATE DETECTOR (YOLO-based)
Better accuracy than simple method
"""

import cv2
import numpy as np
import easyocr
import os
from ultralytics import YOLO
from matplotlib import pyplot as plt

class AdvancedPlateDetector:
    def __init__(self):
        """
        Initialize detector with YOLO and OCR
        """
        print("🚀 Initializing Advanced Detector...")
        
        # Check if YOLO model exists
        if not os.path.exists('yolov8n.pt'):
            print("❌ YOLO model not found!")
            print("💡 Please run: python download_yolo_model.py")
            exit(1)
        
        print("📦 Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        print("📦 Loading OCR (15-20 seconds)...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        
        print("✅ All systems ready!")
    
    def detect_vehicles(self, image):
        """
        YOLO se vehicles detect karo
        """
        # YOLO prediction
        results = self.yolo_model.predict(image, conf=0.3, verbose=False)
        
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Class ID check karo (car, truck, bus, etc.)
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Vehicles: car(2), truck(7), bus(5), motorbike(3)
                if class_id in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'confidence': confidence
                    })
        
        return vehicles
    
    def find_plate_in_vehicle(self, vehicle_img):
        """
        Vehicle image me number plate search karo
        """
        # Grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(gray, 30, 200)
        
        # Contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        # Find rectangular contour
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Aspect ratio check (plates are usually wider than tall)
                aspect_ratio = w / float(h)
                
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 20:
                    return (x, y, w, h), vehicle_img[y:y+h, x:x+w]
        
        return None, None
    
    def read_plate_text(self, plate_img):
        """
        OCR se plate text read karo
        """
        if plate_img is None or plate_img.size == 0:
            return "NO_PLATE_FOUND"
        
        # Preprocessing
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        results = self.ocr_reader.readtext(thresh)
        
        text = ""
        for result in results:
            text += result[1] + " "
        
        return text.strip().replace(" ", "") if text.strip() else "NO_TEXT"
    
    def detect_and_read(self, image_path):
        """
        Complete detection pipeline
        """
        print(f"\n📸 Processing: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Error: Image not found!")
            return None
        
        print("✅ Image loaded")
        original = img.copy()
        
        # Step 1: Detect vehicles
        print("🔍 Step 1: Detecting vehicles...")
        vehicles = self.detect_vehicles(img)
        
        if not vehicles:
            print("⚠️ No vehicles detected, trying direct plate detection...")
            # Directly search for plate in whole image
            bbox, plate_img = self.find_plate_in_vehicle(img)
            
            if plate_img is not None:
                text = self.read_plate_text(plate_img)
                x, y, w, h = bbox
                
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(original, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                return original, [{'text': text, 'bbox': bbox, 
                                  'plate_image': plate_img}]
            else:
                print("❌ No number plate found!")
                return original, []
        
        print(f"✅ Found {len(vehicles)} vehicle(s)")
        
        # Step 2: Find plates in each vehicle
        print("🔍 Step 2: Searching for number plates...")
        
        detected_plates = []
        
        for i, vehicle in enumerate(vehicles, 1):
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_img = img[y1:y2, x1:x2]
            
            print(f"   Vehicle {i}: {vehicle['class']} "
                  f"(confidence: {vehicle['confidence']:.2f})")
            
            # Find plate
            plate_bbox, plate_img = self.find_plate_in_vehicle(vehicle_img)
            
            if plate_img is not None:
                # Read text
                text = self.read_plate_text(plate_img)
                
                # Adjust coordinates to original image
                px, py, pw, ph = plate_bbox
                abs_x = x1 + px
                abs_y = y1 + py
                
                # Draw vehicle bbox (blue)
                cv2.rectangle(original, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw plate bbox (green)
                cv2.rectangle(original, (abs_x, abs_y), 
                            (abs_x+pw, abs_y+ph), (0, 255, 0), 3)
                
                # Text
                cv2.putText(original, text, (abs_x, abs_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                print(f"   ✅ Plate detected: {text}")
                
                detected_plates.append({
                    'text': text,
                    'bbox': (abs_x, abs_y, pw, ph),
                    'plate_image': plate_img,
                    'vehicle': vehicle['class']
                })
            else:
                print(f"   ⚠️ No plate found in this vehicle")
        
        return original, detected_plates
    
    def process_image(self, image_path):
        """
        Process single image and save results
        """
        result_img, plates = self.detect_and_read(image_path)
        
        if result_img is None:
            return
        
        # Save results
        os.makedirs('data/output', exist_ok=True)
        
        output_path = f'data/output/advanced_{os.path.basename(image_path)}'
        cv2.imwrite(output_path, result_img)
        
        print(f"\n📊 RESULTS:")
        print(f"   Detected {len(plates)} plate(s)")
        
        for i, plate in enumerate(plates, 1):
            print(f"   {i}. {plate['text']} ({plate.get('vehicle', 'Unknown')})")
            
            # Save individual plates
            plate_path = f'data/output/plate_{i}_{os.path.basename(image_path)}'
            cv2.imwrite(plate_path, plate['plate_image'])
        
        print(f"\n💾 Results saved:")
        print(f"   Full image: {output_path}")
        print(f"   Check data/output/ for cropped plates")
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected {len(plates)} Number Plate(s)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'data/output/viz_{os.path.basename(image_path)}', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return result_img, plates


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    print("="*50)
    print("🚗 ADVANCED NUMBER PLATE DETECTOR")
    print("="*50)
    
    # Initialize detector
    detector = AdvancedPlateDetector()
    
    print("\n📁 Available images:")
    image_folder = 'data/images'
    
    if os.path.exists(image_folder):
        images = [f for f in os.listdir(image_folder) 
                 if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
        
        if images:
            for i, img in enumerate(images, 1):
                print(f"   {i}. {img}")
            
            choice = input(f"\nSelect image (1-{len(images)}) or 'all': ").strip()
            
            if choice.lower() == 'all':
                # Process all images
                print("\n📦 Batch processing mode...")
                for img in images:
                    img_path = os.path.join(image_folder, img)
                    try:
                        detector.process_image(img_path)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    print("\n" + "-"*50)
            else:
                # Process single image
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(images):
                        img_path = os.path.join(image_folder, images[idx])
                        detector.process_image(img_path)
                    else:
                        print("❌ Invalid choice!")
                except:
                    print("❌ Invalid input!")
        else:
            print("❌ No images found in data/images/")
    else:
        print("❌ data/images/ folder not found!")
    
    print("\n✨ Processing complete!")