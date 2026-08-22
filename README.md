# 🚗 Vehicle Number Plate Detection

> An AI-powered Automatic License Plate Recognition (ALPR) system that detects vehicle number plates from images and videos and extracts the plate text using Computer Vision, YOLOv8, and OCR.

This project provides **two detection engines**:

* 🔹 **Simple Detector** using traditional OpenCV techniques
* 🔹 **Advanced Detector** using YOLOv8 for vehicle detection combined with OpenCV and EasyOCR

It also includes an interactive **Streamlit web application** for uploading and processing images or videos.

---

## 🌟 Project Overview

Vehicle number plate recognition is an important Computer Vision application with use cases in:

* Smart parking systems
* Traffic monitoring
* Toll collection systems
* Security and surveillance
* Vehicle tracking
* Automated access control

This project detects a vehicle's license plate and extracts the text from it using an end-to-end detection pipeline.

The repository supports both a lightweight traditional Computer Vision approach and a more advanced YOLOv8-assisted approach.

---

# ✨ Key Features

* 📷 Detect number plates from images
* 🎥 Process vehicle videos frame by frame
* 🤖 YOLOv8-based vehicle detection
* 👁️ Traditional OpenCV-based plate detection
* 🔤 OCR-based number plate text extraction using EasyOCR
* 🌐 Interactive Streamlit web application
* 📁 Support for JPG, JPEG, PNG, MP4, AVI, and MOV files
* 🔍 Contour detection and rectangular plate filtering
* 📊 Step-by-step diagnostic visualizations
* 📝 CSV export for detected number plates in videos
* ⏱️ Timestamp and bounding box logging
* 🚀 Automated YOLO model downloading
* 🧪 Automated sample test-data downloading

---

# 🧠 Detection Approaches

## 1️⃣ Simple Detector: Traditional Computer Vision

The simple detector uses OpenCV image processing techniques to identify possible number plate regions.

### Pipeline

```text id="vt1cv"
Input Image
     │
     ▼
Convert to Grayscale
     │
     ▼
Bilateral Filtering
     │
     ▼
Canny Edge Detection
     │
     ▼
Find Contours
     │
     ▼
Filter Rectangular Regions
     │
     ▼
Crop Number Plate
     │
     ▼
Image Enhancement
     │
     ▼
EasyOCR
     │
     ▼
Extracted Number Plate Text
```

### Best For

* High-contrast images
* Simple backgrounds
* Lightweight systems
* Learning traditional Computer Vision
* Systems without heavy model requirements

---

# 2️⃣ Advanced Detector: YOLOv8 + OpenCV + OCR

The advanced detector first identifies the vehicle using YOLOv8.

Instead of searching the entire image for a number plate, the system focuses the search inside the detected vehicle region.

### Pipeline

```text id="vt2yolo"
Input Image / Video
          │
          ▼
YOLOv8 Vehicle Detection
          │
          ▼
Detect Vehicle Bounding Box
          │
          ▼
Crop Vehicle Region
          │
          ▼
OpenCV Plate Detection
          │
          ▼
Contour Analysis
          │
          ▼
Extract Plate Region
          │
          ▼
Image Enhancement
          │
          ▼
EasyOCR
          │
          ▼
Detected License Plate Text
```

### Best For

* Complex backgrounds
* Multiple vehicles
* More challenging scenes
* Images where full-image contour detection may fail

---

# 🏗️ System Architecture

```text id="vta31"
                 ┌─────────────────────┐
                 │   Image / Video     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Choose Detector   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐
     │ Simple Detector │        │ Advanced YOLOv8 │
     │     OpenCV      │        │    Detector     │
     └────────┬────────┘        └────────┬────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Number Plate Region │
                 │      Detection      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Enhancement   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      EasyOCR        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Plate Number Output │
                 └─────────────────────┘
```

---

# 🛠️ Tech Stack

| Technology  | Purpose                          |
| ----------- | -------------------------------- |
| Python      | Core programming language        |
| OpenCV      | Image and video processing       |
| YOLOv8      | Vehicle detection                |
| Ultralytics | YOLO model implementation        |
| EasyOCR     | Number plate text recognition    |
| Streamlit   | Interactive web application      |
| NumPy       | Numerical operations             |
| Pandas      | CSV and structured data handling |

---

# 📂 Project Structure

```text id="vtstructure"
Vehicle-number-plate-detection/
│
├── advanced_detector.py
├── simple_detector.py
├── web_app.py
│
├── download_yolo_model.py
├── download_test_data.py
│
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

## File Description

### `simple_detector.py`

Implements traditional Computer Vision techniques for detecting vehicle number plates using:

* Grayscale conversion
* Bilateral filtering
* Canny edge detection
* Contour detection
* Rectangular region filtering
* Plate cropping
* OCR text extraction

---

### `advanced_detector.py`

Implements an advanced detection pipeline using:

* YOLOv8 vehicle detection
* Vehicle bounding boxes
* Region-based plate detection
* OpenCV image processing
* EasyOCR text recognition

---

### `web_app.py`

Provides an interactive Streamlit web interface where users can:

* Upload images
* Upload videos
* Process vehicle files
* View detection results
* Adjust video frame processing
* Download generated outputs

---

### `download_yolo_model.py`

Automatically downloads the required pretrained YOLO model.

---

### `download_test_data.py`

Downloads sample vehicle images for testing the application.

---

### `requirements.txt`

Contains all required Python libraries.

---

### `packages.txt`

Contains system-level dependencies required for deployment environments where applicable.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash id="vtclone"
git clone https://github.com/MdShahzad786-AI/Vehicle-number-plate-detection.git
```

## 2. Navigate to the Project Directory

```bash id="vtcd"
cd Vehicle-number-plate-detection
```

## 3. Create a Virtual Environment

```bash id="vtvenv"
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```bash id="vtwin"
venv\Scripts\activate
```

### macOS/Linux

```bash id="vtlinux"
source venv/bin/activate
```

## 5. Install Dependencies

```bash id="vtinstall"
pip install -r requirements.txt
```

---

# 📥 Download YOLO Model

Run:

```bash id="vtyolo"
python download_yolo_model.py
```

This downloads the pretrained YOLO model required by the advanced detection system.

---

# 🧪 Download Sample Test Data

Run:

```bash id="vttest"
python download_test_data.py
```

This downloads sample vehicle images for testing.

---

# ▶️ Running the Project

## Option 1: Streamlit Web Application

Run:

```bash id="vtstreamlit"
streamlit run web_app.py
```

The application will open in your browser.

From the web interface, you can upload:

* `.jpg`
* `.jpeg`
* `.png`
* `.mp4`
* `.avi`
* `.mov`

---

## Option 2: Run the Simple Detector

```bash id="vtsimple"
python simple_detector.py
```

This uses the traditional OpenCV-based detection pipeline.

---

## Option 3: Run the Advanced Detector

```bash id="vtadvanced"
python advanced_detector.py
```

This uses YOLOv8-assisted vehicle detection combined with OpenCV and OCR.

---

# 📊 Output

The system can generate outputs such as:

* Detected vehicle images
* Number plate crops
* OCR results
* Annotated images
* Diagnostic visualizations
* Processed video results
* CSV files containing detected plates
* Detection timestamps
* Plate bounding box information

---

# 🔍 Example Processing Flow

```text id="vtexample"
Input:
Vehicle Image / Video

        ↓

Vehicle Detection
        ↓

Number Plate Region Detection
        ↓

Image Preprocessing
        ↓

Plate Cropping
        ↓

OCR Text Recognition
        ↓

Output:

┌──────────────────────────────┐
│ Vehicle Number: JH01AB1234   │
│ Timestamp: 00:00:05          │
│ Detection Coordinates: (...) │
└──────────────────────────────┘
```

---

# 📸 Screenshots

Add screenshots of your project here.

Create an `assets` folder:

```text id="vtassets"
assets/
├── web_app.png
├── detection_result.png
├── advanced_detection.png
└── diagnostic_output.png
```

Then add them to the README:

```markdown id="vtimg"
## Web Application

![Web Application](assets/web_app.png)

## Detection Result

![Detection Result](assets/detection_result.png)

## Advanced YOLO Detection

![Advanced Detection](assets/advanced_detection.png)
```

> 📌 Adding real screenshots will make your GitHub project much stronger for recruiters and hiring managers.

---

# 🧪 Example Use Cases

This project can be extended for:

* 🚗 Smart parking systems
* 🛣️ Traffic monitoring
* 🏢 Vehicle access control
* 🚓 Security and surveillance
* 🅿️ Automated parking management
* 💳 Toll collection systems
* 📹 CCTV-based vehicle tracking

---

# ⚠️ Current Limitations

The current system may have reduced accuracy in situations such as:

* Low-resolution images
* Blurry videos
* Poor lighting
* Extreme camera angles
* Occluded number plates
* Motion blur
* Very small number plates
* Non-standard plate formats

OCR accuracy can also depend on the image quality and visibility of characters.

---

# 🔮 Future Improvements

* [ ] Custom-trained license plate detection model
* [ ] Support for real-time webcam detection
* [ ] License plate tracking across video frames
* [ ] Improved OCR preprocessing
* [ ] Confidence score display
* [ ] Support for multiple OCR engines
* [ ] Database integration
* [ ] Vehicle and plate detection API using FastAPI
* [ ] Docker support
* [ ] Cloud deployment
* [ ] Real-time CCTV integration
* [ ] Vehicle make and model detection
* [ ] Stream processing for live video feeds

---

# 🧠 Key Learning Outcomes

Through this project, I gained practical experience with:

* Computer Vision
* OpenCV
* Object Detection
* YOLOv8
* OCR
* EasyOCR
* Image preprocessing
* Edge detection
* Contour detection
* Video frame processing
* Bounding boxes
* Streamlit
* AI application development

---

# 👨‍💻 Author

**Mohammed Shahzad**

Aspiring **AI/ML Engineer** passionate about building practical applications using:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Deep Learning
* Generative AI

### GitHub

https://github.com/MdShahzad786-AI

---

# ⭐ Support

If you found this project useful, please consider giving it a **star ⭐**.

It helps others discover the project and motivates me to continue building and sharing more AI projects.

---

# 📄 License

This project is open source and available for educational and personal use.
