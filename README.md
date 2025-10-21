
# 🕵️ Deepfake Image Detection using OpenCV

## 🌟 Overview

This is a **fun mini-project** designed to detect **deepfake or AI-generated images** using **OpenCV** and basic computer vision techniques.
With the increasing spread of deepfakes — hyper-realistic AI-generated faces — this project aims to **analyze facial inconsistencies**, **pixel artifacts**, and **visual cues** to differentiate **real images from fake ones**.

The project provides a simple **Flask-based web interface** (optional) or can be run as a **standalone Python script**, making it suitable for learning, experimentation, and awareness purposes.

---

## 🎯 Project Objective

> “To build a simple yet effective system that detects and flags deepfaked or AI-generated images using OpenCV and image analysis techniques.”

The main goal is **educational** — to demonstrate how computer vision can help identify forged media content using accessible methods and libraries.

---

## 🚀 Features

* 🧠 Detects possible deepfakes based on facial feature inconsistencies
* 👁️ Highlights unusual artifacts in the image (edges, lighting, asymmetry)
* 📸 Works on any image containing human faces
* ⚙️ Uses OpenCV for feature extraction and anomaly detection
* 💾 Option to upload and test images via Flask web interface
* 🧮 Simple, lightweight, and easy to understand — perfect for beginners

---

## 🧰 Tech Stack

| Component                  | Technology                                     |
| -------------------------- | ---------------------------------------------- |
| **Programming Language**   | Python                                         |
| **Libraries Used**         | OpenCV, NumPy, Matplotlib, DeepFace (optional) |
| **Optional Web Framework** | Flask                                          |
| **Purpose**                | Deepfake Detection via Visual Analysis         |

---

## 📂 Project Structure

```
deepfake_detector/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html
├── deepfake_detector.py
├── app.py                # Flask Web App (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/deepfake-image-detector.git
cd deepfake-image-detector
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # macOS/Linux
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

Your `requirements.txt` might include:

```
opencv-python
numpy
matplotlib
deepface
flask
```

### 4️⃣ Run the Script

If running standalone:

```bash
python deepfake_detector.py
```

Or to run the Flask web app:

```bash
python app.py
```

Then visit `http://127.0.0.1:5000/` in your browser.

---

## 🧠 How It Works

1. **Image Input**
   User uploads or selects an image containing a human face.

2. **Face Detection (OpenCV)**
   The Haar Cascade or DNN face detector identifies facial regions.

3. **Feature Analysis**
   The system analyzes:

   * Facial symmetry (left vs. right comparison)
   * Edge sharpness and pixel-level artifacts
   * Inconsistent lighting or blurring
   * Eye and mouth region texture differences

4. **Deepfake Probability Score (Optional)**
   If `DeepFace` or other pretrained embeddings are used, the system can estimate a “realness” confidence score.

5. **Result Display**
   The analyzed image is displayed with a **"Likely Real"** or **"Likely Deepfake"** label.

---

## 🧩 Example Code Snippet

```python
import cv2
import numpy as np

def detect_deepfake(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        edges = cv2.Canny(face, 100, 200)
        blur_score = cv2.Laplacian(face, cv2.CV_64F).var()

        if blur_score < 150:  # threshold can be tuned
            label = "Possibly Deepfake"
            color = (0, 0, 255)
        else:
            label = "Likely Real"
            color = (0, 255, 0)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Deepfake Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## 🧪 Example Results

| Input Image                                               | Detected Output      |
| --------------------------------------------------------- | -------------------- |
| Real Human Face                                           | ✅ Likely Real        |
| AI-generated Face (e.g., from ThisPersonDoesNotExist.com) | ⚠️ Possibly Deepfake |

> *(You can include actual screenshots in `/samples` folder.)*

---

## 🧬 Optional Advanced Features

You can enhance the project using:

* **DeepFace** embeddings comparison for real vs. synthetic faces
* **Frequency spectrum analysis** to detect GAN-generated patterns
* **CNN-based classifier** trained on real vs. fake datasets (Celeb-DF, DFDC)
* **Eye blink detection** in videos (for deepfake video analysis)

---

## 🧱 Requirements

* Python ≥ 3.8
* OpenCV ≥ 4.5
* NumPy ≥ 1.21
* Matplotlib ≥ 3.5
* Flask ≥ 2.2 *(if using web interface)*

---

## 🌐 Deployment

You can deploy the Flask web version using:

* [Render](https://render.com/)
* [Railway](https://railway.app/)
* [Heroku](https://www.heroku.com/)
* Localhost or Docker container

---

## 🧩 Applications

* 🔍 Awareness demonstration for detecting manipulated media
* 🧠 Educational tool for learning computer vision basics
* 🧪 Baseline framework for more advanced deepfake detection research
* 💬 Fun project for AI enthusiasts to explore OpenCV and image analysis

---

## 🛠️ Future Enhancements

* 🤖 Train a CNN model on real vs fake datasets
* 🎥 Add support for deepfake **video** detection
* 📈 Introduce explainable AI visualization (heatmaps for fake areas)
* ☁️ Deploy as a public web app with REST API
* 🧬 Integrate with Hugging Face pretrained detectors

---

## 👨‍💻 Author

**Dhanush S J**
💼 Computer Science and Engineering(AI/ML) Undergraduate
🚀 Enthusiast in AI, Computer Vision, and Ethical Deepfake Detection
🔗 [GitHub](https://github.com/DHANUSHMURTHY11) | [LinkedIn](www.linkedin.com/in/dhanush-murthy)

---


## ⭐ Acknowledgements

* [OpenCV](https://opencv.org/) — for image analysis and feature extraction
* [DeepFace](https://github.com/serengil/deepface) — optional deepfake analysis
* [NumPy](https://numpy.org/) — numerical computations
* [Matplotlib](https://matplotlib.org/) — visualization and result plotting
* [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/) — for testing fake images

## Original Image

<img width="509" height="632" alt="Screenshot_2025-01-05_220234" src="https://github.com/user-attachments/assets/051a2303-c52f-4d96-9fc6-a6ae4a8f14a9" />


## Deep fake image 

<img width="506" height="631" alt="Screenshot_2025-01-05_220240" src="https://github.com/user-attachments/assets/ea1deb48-29d3-4047-9bc1-70eccc49221c" />




