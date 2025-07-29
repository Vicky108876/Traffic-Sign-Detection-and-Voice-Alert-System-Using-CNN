# 🚦 Traffic Sign Detection and Voice Alert System Using CNN

This project implements a **real-time traffic sign detection and voice alert system** using **OpenCV**, **Convolutional Neural Networks (CNN)**, and **pyttsx3** for speech output. The system is designed to assist drivers by detecting traffic signs from live video feed and providing immediate voice feedback.

---

## 📌 Features

- 🧠 **CNN-based Model**: Detects and classifies traffic signs accurately.
- 🎥 **Real-Time Detection**: Uses OpenCV to capture video and identify traffic signs in real-time.
- 🔊 **Voice Alerts**: Converts detected traffic signs into audio alerts using `pyttsx3`.
- 📊 **User-Friendly Output**: Display of prediction results on screen alongside visual bounding boxes.

---

## 🛠️ Technologies Used

- **Python 3.11**
- **OpenCV** – For image capture and preprocessing
- **TensorFlow / Keras** – For building and training the CNN model
- **pyttsx3** – For converting detected sign names to speech
- **NumPy & Pandas** – For data manipulation
- **Matplotlib** – For visualizing model performance

---
## 🗂️ Dataset Used

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which includes 43 classes of traffic signs for training and testing the CNN model.

Datset  :https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.11 (recommended)

### 📦 Install Required Packages

Run the following command to install dependencies:

```bash
pip install opencv-python numpy keras tensorflow pyttsx3
---
#🧠 Step 1: Train the Model

### You must first train the model using the dataset.

@Download the GTSRB dataset and structure it appropriately.

Run the training script:

python Training_File.py
#### ✅ After successful training, a file named model_trained.p will be saved in the project directory.

#🛑 Step 2: Run Traffic Sign Detection
Once training is complete and model_trained.p is generated:

python TrafficSign_Main.py
