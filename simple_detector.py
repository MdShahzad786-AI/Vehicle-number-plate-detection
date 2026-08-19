"""
SIMPLE NUMBER PLATE DETECTOR
Ye code BINA TRAINING ke kaam karega!
Traditional Computer Vision use karega
"""

import cv2
import numpy as np
import easyocr
import os
from matplotlib import pyplot as plt

print("🚀 Starting Number Plate Detector...")
print("📦 Loading OCR (15-20 seconds lagega pehli baar)...")

# OCR Reader initialize karo
reader = easyocr.Reader(['en'], gpu=False)

print("✅ OCR Loaded Successfully!")

def detect_plate(image_path):
    """
    Main function - Number plate detect karega
    """
    print(f"\n📸 Processing: {image_path}")
    
    # Step 1: Image load karo
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Error: Image nahi mili - {image_path}")
        return
    
    print("✅ Image loaded successfully!")
    print(f"   Size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Original image ka copy rakho
    original = img.copy()
    
    # Step 2: Image ko Gray (black & white) me convert karo
    print("🔄 Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Noise remove karo (image ko smooth banao)
    print("🔄 Removing noise...")
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Step 4: Edges detect karo (outline nikalo)
    print("🔄 Detecting edges...")
    edged = cv2.Canny(gray, 30, 200)
    
    # Step 5: Contours find karo (shapes find karo)
    print("🔄 Finding contours...")
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sabse bade 10 contours lo
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    print(f"   Found {len(contours)} potential regions")
    
    # Step 6: Rectangle shape find karo (number plate usually rectangular hoti hai)
    plate_contour = None
    
    for contour in contours:
        # Contour ko approximate karo
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        # Agar 4 corners hai (rectangle) to ye number plate ho sakti hai
        if len(approx) == 4:
            plate_contour = approx
            print("✅ Found rectangular region (potential number plate)!")
            break
    
    if plate_contour is None:
        print("❌ No rectangular region found!")
        print("💡 Tip: Try a different image with clear number plate")
        return
    
    # Step 7: Number plate ko crop karo
    print("🔄 Extracting number plate region...")
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = original[y:y+h, x:x+w]
    
    # Step 8: OCR se text read karo
    print("🔄 Reading text with OCR (may take 5-10 seconds)...")
    
    # Plate image ko process karo for better OCR
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # OCR apply karo
    ocr_results = reader.readtext(plate_gray)
    
    # Text extract karo
    plate_text = ""
    for detection in ocr_results:
        text = detection[1]
        plate_text += text + " "
    
    plate_text = plate_text.strip()
    
    if not plate_text:
        plate_text = "NO_TEXT_DETECTED"
        print("⚠️ Warning: Could not read text from plate")
    else:
        print(f"✅ Detected Text: {plate_text}")
    
    # Step 9: Result image pe draw karo
    result_img = original.copy()
    
    # Green rectangle draw karo
    cv2.drawContours(result_img, [plate_contour], -1, (0, 255, 0), 3)
    
    # Text likhao
    cv2.putText(result_img, plate_text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Step 10: Results save aur display karo
    print("💾 Saving results...")
    
    # Output folder check karo
    os.makedirs('data/output', exist_ok=True)
    
    # Result save karo
    output_path = f'data/output/result_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, result_img)
    
    # Cropped plate save karo
    plate_output = f'data/output/plate_{os.path.basename(image_path)}'
    cv2.imwrite(plate_output, plate_img)
    
    print(f"✅ Results saved!")
    print(f"   Full image: {output_path}")
    print(f"   Plate only: {plate_output}")
    
    # Display results using matplotlib
    print("🖼️ Displaying results...")
    
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('1. Original Image')
    plt.axis('off')
    
    # Grayscale
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('2. Grayscale')
    plt.axis('off')
    
    # Edges
    plt.subplot(2, 3, 3)
    plt.imshow(edged, cmap='gray')
    plt.title('3. Edge Detection')
    plt.axis('off')
    
    # Result with detection
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'4. Detection Result\n{plate_text}')
    plt.axis('off')
    
    # Cropped plate
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    plt.title('5. Extracted Plate')
    plt.axis('off')
    
    # Plate grayscale (what OCR sees)
    plt.subplot(2, 3, 6)
    plt.imshow(plate_gray, cmap='gray')
    plt.title('6. Plate (OCR Input)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_output = f'data/output/visualization_{os.path.basename(image_path)}'
    plt.savefig(viz_output, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_output}")
    
    plt.show()
    
    print("\n" + "="*50)
    print("🎉 DETECTION COMPLETE!")
    print("="*50)
    print(f"📝 Detected Number: {plate_text}")
    print(f"📁 Check 'data/output/' folder for results")
    print("="*50)


def process_all_images():
    """
    data/images folder me saari images process karo
    """
    print("\n" + "="*50)
    print("📁 BATCH PROCESSING MODE")
    print("="*50)
    
    image_folder = 'data/images'
    
    # Check if folder exists
    if not os.path.exists(image_folder):
        print(f"❌ Error: '{image_folder}' folder not found!")
        print("💡 Please create the folder and add images")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
    
    if not image_files:
        print(f"❌ No images found in '{image_folder}'")
        print("💡 Please add some images (.jpg, .png) to test")
        return
    
    print(f"✅ Found {len(image_files)} images to process")
    print("-"*50)
    
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")
        img_path = os.path.join(image_folder, img_file)
        
        try:
            detect_plate(img_path)
            results.append({'file': img_file, 'status': 'SUCCESS'})
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            results.append({'file': img_file, 'status': 'FAILED'})
    
    # Summary
    print("\n" + "="*50)
    print("📊 PROCESSING SUMMARY")
    print("="*50)
    
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    
    print(f"✅ Successful: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved in: data/output/")
    print("="*50)


# ==========================================
# MAIN PROGRAM - YE PART RUN HOGA
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚗 NUMBER PLATE DETECTION SYSTEM")
    print("="*50)
    print("Choose an option:")
    print("1. Process single image")
    print("2. Process all images in data/images/")
    print("="*50)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Single image mode
        print("\n📁 Available images in data/images/:")
        
        image_folder = 'data/images'
        if os.path.exists(image_folder):
            images = [f for f in os.listdir(image_folder) 
                     if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
            
            if images:
                for i, img in enumerate(images, 1):
                    print(f"   {i}. {img}")
                
                img_choice = input(f"\nEnter image number (1-{len(images)}): ").strip()
                
                try:
                    idx = int(img_choice) - 1
                    if 0 <= idx < len(images):
                        image_path = os.path.join(image_folder, images[idx])
                        detect_plate(image_path)
                    else:
                        print("❌ Invalid choice!")
                except:
                    print("❌ Invalid input!")
            else:
                print("❌ No images found!")
                print("💡 Add some images to data/images/ folder")
        else:
            print("❌ data/images/ folder not found!")
            print("💡 Create folder and add images")
    
    elif choice == '2':
        # Batch processing mode
        process_all_images()
    
    else:
        print("❌ Invalid choice! Please enter 1 or 2")
    
    print("\n✨ Thank you for using Number Plate Detector!")