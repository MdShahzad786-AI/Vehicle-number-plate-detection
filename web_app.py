"""
WEB INTERFACE - Number Plate Detection System (Image & Video)
Streamlit GUI with support for Images and Videos using OpenCV, YOLOv8, and EasyOCR.

Run: streamlit run web_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import os
import tempfile
import time
import pandas as pd
from ultralytics import YOLO

# Page setup
st.set_page_config(
    page_title="Number Plate Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Cache OCR reader (load once)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# Cache YOLO model
@st.cache_resource
def load_yolo():
    if os.path.exists('yolov8n.pt'):
        return YOLO('yolov8n.pt')
    return None

def detect_plate_simple(image):
    """
    Simple OpenCV Contour Detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            plate_contour = approx
            break
    
    if plate_contour is None:
        return None, None, None
    
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = image[y:y+h, x:x+w]
    
    return plate_contour, (x, y, w, h), plate_img

def detect_plate_yolo(image, yolo_model):
    """
    YOLOv8 + Contour Plate Detection
    """
    if yolo_model is None:
        return detect_plate_simple(image)
    
    results = yolo_model.predict(image, conf=0.25, verbose=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            # Vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
            if class_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_crop = image[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    contour, bbox, plate_img = detect_plate_simple(vehicle_crop)
                    if plate_img is not None:
                        px, py, pw, ph = bbox
                        abs_bbox = (x1 + px, y1 + py, pw, ph)
                        abs_contour = contour + np.array([x1, y1])
                        return abs_contour, abs_bbox, plate_img
    
    # Fallback to simple detector if YOLO doesn't isolate vehicle plate
    return detect_plate_simple(image)

def read_text_from_plate(plate_img, reader):
    """
    OCR Text Extraction
    """
    if plate_img is None or plate_img.size == 0:
        return "NO_PLATE_FOUND"
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    results = reader.readtext(thresh)
    text = ""
    for result in results:
        text += result[1] + " "
    
    clean_text = "".join([c for c in text if c.isalnum() or c == ' ']).strip().upper()
    return clean_text if clean_text else "NO_TEXT"

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.markdown("<div class='main-header'>🚗 Number Plate Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>AI-Powered Automatic License Plate Recognition (ALPR) for Images & Videos</div>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ System Control")
    detector_type = st.sidebar.radio(
        "Select Detection Engine:",
        ["Advanced (YOLOv8 + OpenCV)", "Simple (OpenCV Contours)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.title("ℹ️ Project Info")
    st.sidebar.info(
        """
        **Number Plate Detection System**
        - **Supported Input:** Images & Videos
        - **Tech Stack:** OpenCV, EasyOCR, YOLOv8, Streamlit
        - **Status:** Local Host Ready
        """
    )
    
    # Load Models
    with st.spinner("Loading OCR and Object Detection models..."):
        reader = load_ocr()
        yolo_model = load_yolo() if "Advanced" in detector_type else None
    
    st.sidebar.success("✅ Models Loaded Successfully!")
    
    # Tabs for Image & Video
    tab_image, tab_video = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    # ----------------------------------------------------
    # TAB 1: IMAGE DETECTION
    # ----------------------------------------------------
    with tab_image:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a vehicle image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Resolution: {image.shape[1]}x{image.shape[0]} px")
            
            if st.button("🔍 Detect Number Plate from Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    if "Advanced" in detector_type and yolo_model is not None:
                        contour, bbox, plate_img = detect_plate_yolo(image, yolo_model)
                    else:
                        contour, bbox, plate_img = detect_plate_simple(image)
                    
                    if plate_img is None or plate_img.size == 0:
                        st.error("❌ No number plate detected in this image.")
                        st.info("💡 Tip: Upload a clearer image where the license plate is facing front or rear directly.")
                    else:
                        plate_text = read_text_from_plate(plate_img, reader)
                        
                        result_img = image.copy()
                        if contour is not None:
                            cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 3)
                        
                        x, y, w, h = bbox
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(result_img, plate_text, (x, max(30, y-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        with col2:
                            st.subheader("✅ Detection Result")
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("📊 Detection Metrics & Extracted Plate")
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric("Detected Plate Text", plate_text)
                        with m_col2:
                            st.metric("Plate Width", f"{w} px")
                        with m_col3:
                            st.metric("Plate Height", f"{h} px")
                        
                        p_col1, p_col2 = st.columns([1, 2])
                        with p_col1:
                            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), caption="Cropped License Plate", use_container_width=True)
                        with p_col2:
                            st.markdown("**Plate Details:**")
                            st.write(f"- **Recognized Text:** `{plate_text}`")
                            st.write(f"- **Bounding Box (X, Y, W, H):** `({x}, {y}, {w}, {h})`")
                            st.write(f"- **Aspect Ratio:** `{w/h:.2f}`")
                            if plate_text != "NO_TEXT" and plate_text != "NO_PLATE_FOUND":
                                st.success("✅ Number plate recognized successfully!")
                            else:
                                st.warning("⚠️ Plate outline detected, but text could not be read cleanly.")
                        
                        # Download buttons
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            _, buffer = cv2.imencode('.jpg', result_img)
                            st.download_button(
                                label="📥 Download Annotated Image",
                                data=buffer.tobytes(),
                                file_name="plate_detection_result.jpg",
                                mime="image/jpeg"
                            )
                        with dl_col2:
                            _, buffer2 = cv2.imencode('.jpg', plate_img)
                            st.download_button(
                                label="📥 Download Cropped Plate",
                                data=buffer2.tobytes(),
                                file_name="cropped_plate.jpg",
                                mime="image/jpeg"
                            )
        else:
            st.info("👆 Upload an image above to start detecting license plates.")
            
    # ----------------------------------------------------
    # TAB 2: VIDEO DETECTION
    # ----------------------------------------------------
    with tab_video:
        st.header("Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a traffic or vehicle video (MP4, AVI, MOV, MKV)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        frame_skip = st.slider("Frame Skip Step (Speed up processing):", min_value=1, max_value=10, value=3, help="Process every Nth frame to increase speed.")
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            tfile.close()
            
            st.video(video_path)
            
            if st.button("🎥 Start Video Detection", type="primary", use_container_width=True):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.info(f"Video Stats: {total_frames} frames, {fps:.1f} FPS, {width}x{height} resolution")
                
                # Output video setup
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_placeholder = st.empty()
                
                detections_log = []
                frame_count = 0
                processed_count = 0
                
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    if frame_count % frame_skip == 0 or frame_count == 1:
                        processed_count += 1
                        
                        if "Advanced" in detector_type and yolo_model is not None:
                            contour, bbox, plate_img = detect_plate_yolo(frame, yolo_model)
                        else:
                            contour, bbox, plate_img = detect_plate_simple(frame)
                        
                        if plate_img is not None and plate_img.size > 0:
                            plate_text = read_text_from_plate(plate_img, reader)
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            cv2.putText(frame, plate_text, (x, max(30, y-10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            timestamp_sec = frame_count / fps
                            detections_log.append({
                                "Frame": frame_count,
                                "Timestamp (s)": round(timestamp_sec, 2),
                                "Plate Text": plate_text,
                                "Position (X,Y)": f"({x}, {y})",
                                "Size (WxH)": f"{w}x{h}"
                            })
                        
                        preview_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Live Preview - Frame {frame_count}/{total_frames}", use_container_width=True)
                    
                    out_writer.write(frame)
                    
                    progress_pct = int((frame_count / max(1, total_frames)) * 100)
                    progress_bar.progress(min(100, progress_pct))
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress_pct}%)...")
                
                cap.release()
                out_writer.release()
                
                elapsed = time.time() - start_time
                status_text.success(f"🎉 Video Processing Complete in {elapsed:.2f} seconds!")
                
                st.markdown("---")
                st.subheader("📊 Detected Number Plates Log")
                
                if detections_log:
                    df = pd.DataFrame(detections_log)
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Detection Report (CSV)",
                        data=csv,
                        file_name="video_plate_detections.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No number plates detected in the uploaded video.")
                
                # Video Download
                with open(out_path, 'rb') as vf:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=vf.read(),
                        file_name="processed_plate_detection.mp4",
                        mime="video/mp4"
                    )
                
                # Clean temp files
                try:
                    os.remove(video_path)
                    os.remove(out_path)
                except Exception:
                    pass
        else:
            st.info("👆 Upload a video above to test vehicle number plate detection in motion.")

if __name__ == "__main__":
    main()