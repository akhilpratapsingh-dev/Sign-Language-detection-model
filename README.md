# 🤟 Real-Time Hand Sign Language Recognition

This repository contains the code for a real-time system that recognizes American Sign Language (ASL) letters (A-Z) from live video input using hand gestures. It leverages the **Mediapipe Hands** solution for robust hand landmark detection and a custom-trained **Machine Learning model** for classification.

## ✨ Features

* **Real-Time Detection:** Processes video frames from a webcam to detect hand gestures instantly.
* **Hand Landmark Tracking:** Uses **Mediapipe** to accurately locate **21 key hand landmarks**.
* **Model Integration:** Loads a pre-trained model (from `model.p`) to classify the gesture.
* **Visual Feedback:** Draws hand landmarks, a bounding box around the detected hand, and displays the predicted sign (ASL letter) on the video feed.

---

## ⚙️ Prerequisites

To run this project, you'll need the following installed:

* **Python 3.x**
* **A Webcam**
* **Required Python Libraries:**
    * `opencv-python`
    * `mediapipe`
    * `numpy`
    * `scikit-learn`
