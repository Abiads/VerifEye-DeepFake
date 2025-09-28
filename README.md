# 🔍 VerifEye - DeepFake Detection System

<div align="center">

![DeepFake Detection](https://img.shields.io/badge/DeepFake-Detection-red?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Advanced AI-powered system for detecting deepfake videos using CNN-RNN hybrid architecture**

[📹 Demo Videos](#-demo) • [🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [👥 Team](#-team)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [📹 Demo](#-demo)
- [⚠️ Why DeepFake Detection Matters](#️-why-deepfake-detection-matters)
- [🎯 Project Objectives](#-project-objectives)
- [🏗️ System Architecture](#️-system-architecture)
- [🧠 Models & Performance](#-models--performance)
- [🚀 Quick Start](#-quick-start)
- [💻 Installation & Usage](#-installation--usage)
- [🛠️ Technologies Used](#️-technologies-used)
- [📈 Results & Accuracy](#-results--accuracy)
- [👥 Team](#-team)
- [📄 License](#-license)

---


## 🎯 Overview

**VerifEye** is a cutting-edge deep learning system designed to detect and classify deepfake videos with high accuracy. Our system combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to analyze video frames and identify subtle artifacts that distinguish real videos from AI-generated deepfakes.

### What are DeepFakes?

DeepFakes are sophisticated AI-generated images or videos that have been manipulated to replace one person's face with another's. They use advanced techniques like:
- **Generative Adversarial Networks (GANs)** for realistic face swapping
- **Autoencoders** for seamless video manipulation
- **Neural texture synthesis** for enhanced realism

Our system addresses the growing threat of malicious deepfake content in digital media.

## 📹 Demo

### 🎬 Main Demonstration
**Complete System Showcase**

<div align="center">

**DeepFake Detection Demo Video** 🎥

<video width="800" controls>
  <source src="https://github.com/Abiads/VerifEye-DeepFake/blob/1110ddd5ed056dd1f10411abde87a5a2b9c3f31a/DeepFake-Detection_v720P.mp4" type="video/mp4">
  <source src="./DeepFake-Detection_v720P.mp4" type="video/mp4">
  <source src="DeepFake-Detection_v720P.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Watch our system analyze video frames and classify them as real or fake with **85%+ accuracy***

</div>

### 🧪 Sample Test Video
**Example Input for Testing**

<div align="center">

**Sample Test Video** 🎬

<video width="600" controls>
  <source src="https://github.com/Abiads/VerifEye-DeepFake/blob/1aeae5d9536a3c804438caf938232b7b240e46a2/Deploy/static/upload/adohikbdaz.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Use this sample to test the system's detection capabilities*

</div>

### 🔗 Direct Links
- **[📺 Full Demo Video (GitHub)](https://github.com/Abiads/VerifEye-DeepFake/blob/1110ddd5ed056dd1f10411abde87a5a2b9c3f31a/DeepFake-Detection_v720P.mp4)** - GitHub permalink
- **[📺 Full Demo Video (Local)](./DeepFake-Detection_v720P.mp4)** - Local file path
- **[🧪 Test Sample Video (GitHub)](https://github.com/Abiads/VerifEye-DeepFake/blob/1aeae5d9536a3c804438caf938232b7b240e46a2/Deploy/static/upload/adohikbdaz.mp4)** - GitHub permalink
- **[🧪 Test Sample Video (Local)](./Deploy/static/upload/adohikbdaz.mp4)** - Local file path

### 📁 File Locations
```
📂 Project Root/
├── 📹 DeepFake-Detection_v720P.mp4          (Main demo video)
└── 📂 Deploy/static/upload/
    ├── 📹 adohikbdaz.mp4                     (Sample test video)
    └── 📹 adylbeequz.mp4                     (Additional sample)
```

## ⚠️ Why DeepFake Detection Matters

DeepFake technology poses significant threats across multiple domains:

### 🚨 Critical Impact Areas
- **📰 Fake News & Misinformation** - Spreading false narratives through manipulated videos
- **👨‍💼 Political Manipulation** - Creating fake statements from public figures
- **💰 Financial Fraud** - Identity theft and fraudulent transactions
- **🎭 Entertainment Industry** - Unauthorized use of celebrity likenesses
- **🔒 Security & Privacy** - Non-consensual image/video creation

### 🛡️ Our Solution
Our detection system provides a robust defense mechanism to identify and flag potentially malicious deepfake content before it spreads.
 
## 🎯 Project Objectives

### Primary Goals
1. **🎯 High-Accuracy Detection** - Build a model that processes videos and classifies them as REAL or FAKE with >85% accuracy
2. **⚡ Real-time Processing** - Develop a system capable of analyzing videos in under 1 minute
3. **🌐 Scalable Integration** - Create a deployable solution for social media platforms and content moderation
4. **🔍 Face-Swap Focus** - Specialize in detecting face-swapped deepfake videos

### Technical Approach
- **🧠 Hybrid Architecture** - Combine CNN for feature extraction with RNN for temporal analysis
- **🎨 Frame-by-Frame Analysis** - Detect subtle imperfections in facial landmarks
- **📊 Statistical Learning** - Train models to identify distinguishing features between real and fake content

![image](https://user-images.githubusercontent.com/77656115/206965843-6ac74168-3e31-43d6-9bbf-3e3d25e17522.png)

## 🏗️ System Architecture

### 🔄 Processing Pipeline

```
Video Input → Frame Extraction → Face Detection → Facial Landmark Detection 
    ↓
Feature Extraction (CNN) → Temporal Analysis (RNN) → Classification → Real/Fake Output
```

### 📋 Step-by-Step Process

| Step | Process | Description |
|------|---------|-------------|
| **1** | **Data Loading** | Load video datasets (DFDC, Celeb-DF) |
| **2** | **Frame Extraction** | Extract frames from input videos |
| **3** | **Face Detection** | Identify and crop facial regions |
| **4** | **Landmark Detection** | Locate key facial feature points |
| **5** | **Feature Analysis** | Extract CNN features from face crops |
| **6** | **Temporal Processing** | Analyze frame sequences with RNN |
| **7** | **Classification** | Determine Real vs Fake probability |


### 🔄 Workflow Diagrams

#### Pre-processing Pipeline
![Pre-processing Workflow](https://user-images.githubusercontent.com/77656115/206968030-1e9729e7-8d34-4295-a110-d05ad0ade7bb.png)

#### Prediction Workflow
![Prediction Workflow](https://user-images.githubusercontent.com/77656115/206968272-73db6238-79a0-46a1-ad5b-e651ad002322.png)

## 🧠 Models & Performance

### 🏆 Best Performing Model: **EfficientNetB2 + GRU**

#### ✅ Advantages
- **🎯 High Accuracy**: 85% test accuracy on DFDC dataset
- **⚡ Efficient Processing**: Optimized for both accuracy and speed
- **🔄 Temporal Understanding**: GRU captures video sequence patterns
- **📊 Robust Features**: EfficientNetB2 provides rich feature representations

#### ⚙️ Technical Specifications
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Sparse Categorical Crossentropy
- **Architecture**: CNN (Feature Extraction) + RNN (Classification)
- **Input**: Face-cropped video frames
- **Output**: Real/Fake probability scores

### 📊 Model Comparison

| Model | Architecture | Accuracy | Strengths | Limitations |
|-------|-------------|----------|-----------|-------------|
| **MesoNet** | CNN | ~70% | Pre-trained, fast | Poor video frame detection |
| **ResNet50** | CNN | ~75% | ImageNet weights | Limited temporal analysis |
| **EfficientNetB0** | CNN | ~78% | Lightweight | No sequence modeling |
| **InceptionV3+GRU** | CNN+RNN | ~82% | Good temporal analysis | Multiple face issues |
| **EfficientNetB2+GRU** | CNN+RNN | **~85%** | **Best overall** | Dark background sensitivity |

### 🎯 Performance Metrics

```
Test Accuracy: ~85%
Precision (Fake): High
Recall (Real): High
Processing Time: ~1 minute for 10-second video
```

## 🚀 Quick Start

### ⚡ One-Command Setup
```bash
git clone https://github.com/Abiads/VerifEye-DeepFake.git
cd VerifEye-DeepFake
pip install -r Deploy/requirements.txt
python Deploy/app.py
```

### 🎯 Immediate Testing
1. **Upload a video** through the web interface
2. **Wait for processing** (~1 minute for 10-second video)
3. **View results** with confidence scores

## 💻 Installation & Usage

### 📋 Prerequisites
- **Python**: 3.7 or higher
- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space

### 🔧 Installation Steps

#### 1. Clone Repository
```bash
git clone https://github.com/Abiads/VerifEye-DeepFake.git
cd VerifEye-DeepFake
```

#### 2. Install Dependencies
```bash
cd Deploy
pip install -r requirements.txt
```

#### 3. Run Application
```bash
python app.py
```

#### 4. Access Web Interface
Open your browser and navigate to: `http://localhost:5000`

### 🎮 Usage Instructions

1. **📁 Upload Video**: Select a video file (.mp4, .avi, .mov)
2. **⏳ Processing**: System analyzes frames automatically
3. **📊 Results**: View classification with confidence score
4. **💾 Download**: Save results and analysis report

## 🛠️ Technologies Used

### 🐍 Core Technologies
<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

</div>

### 📚 Libraries & Frameworks

| Category | Technology | Purpose |
|----------|------------|---------|
| **🧠 Deep Learning** | TensorFlow, Keras | Model training and inference |
| **🖼️ Computer Vision** | OpenCV, PIL | Image processing and face detection |
| **🌐 Web Framework** | Flask | Web application interface |
| **📊 Data Processing** | NumPy, Pandas | Data manipulation and analysis |
| **🎨 Frontend** | HTML5, CSS3 | User interface design |
| **📈 Visualization** | Matplotlib, Seaborn | Results visualization |

### 🔧 Development Tools
- **Face Recognition**: dlib, face_recognition
- **Video Processing**: OpenCV, FFmpeg
- **Model Optimization**: TensorFlow Lite
- **API Development**: Flask-RESTful

## 📈 Results & Accuracy

### 🎯 Performance Summary

| Metric | Value | Description |
|--------|-------|-------------|
| **Overall Accuracy** | **85%** | Test accuracy on DFDC dataset |
| **Processing Speed** | **~1 min** | For 10-second 30fps video |
| **Model Size** | **~200MB** | Optimized for deployment |
| **False Positive Rate** | **<5%** | Real videos classified as fake |
| **False Negative Rate** | **<15%** | Fake videos missed |

### 📊 Detailed Analysis

#### ✅ Strengths
- **High Precision**: Excellent at identifying fake videos
- **Fast Processing**: Real-time analysis capabilities
- **Robust Detection**: Works across different video qualities
- **Scalable Architecture**: Easy to deploy and maintain

#### ⚠️ Limitations
- **Multiple Faces**: Performance degrades with multiple people in frame
- **Dark Environments**: Challenging in low-light conditions
- **New Techniques**: May struggle with latest deepfake methods
- **Computational Requirements**: GPU recommended for optimal performance


## 👥 Team

<div align="center">

### 🎯 Project Leader

**Kripanshu Gupta**
- 📧 **Email**: gkripanshustranger@gmail.com
- 📱 **Phone**: +91 7067058400
- 🔗 **LinkedIn**: [Connect with Kripanshu](https://www.linkedin.com/in/kripanshu-gupta-a66349261/)
- 👤 **Role**: Team Lead & Project Coordinator

</div>

### 👨‍💻 Development Team

| Member | Contact | Phone | Gender |
|--------|---------|-------|--------|
| **Aditi Soni** | aditisony027@gmail.com | +91 8982501064 | Female |
| **Jay Kumar** | jaykumar2951@gmail.com | +91 6204776725 | Male |
| **Abhay Gupta** | aviabs098@gmail.com | +91 8959392479 | Male |
| **Avinash Singh** | as3554140@gmail.com | +91 7490913969 | Male |
| **Aditya Vishwakarma** | adityavishwakarmadd@gmail.com | +91 7828439252 | Male |

#### 🌟 Special Recognition
**Abhay Gupta** - LinkedIn: [Professional Profile](https://www.linkedin.com/in/abhay-gupta-197b17264/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 Star this repository if you found it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Abiads/VerifEye-DeepFake?style=social)](https://github.com/Abiads/VerifEye-DeepFake/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Abiads/VerifEye-DeepFake?style=social)](https://github.com/Abiads/VerifEye-DeepFake/network)

**Made with ❤️ by the VerifEye Team**

</div>



