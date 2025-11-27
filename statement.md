🧩 Problem Statement

Communication between hearing-impaired individuals and people who do not understand sign language creates a significant barrier in daily life, education, workplaces, and public services.
Traditional sign language interpreters are not always available, which makes real-time communication difficult.

There is a need for a low-cost, accessible, and real-time system that can understand hand gestures and translate them into readable output.

This project solves this problem by creating a machine learning-based Sign Language Detection System that recognizes A–Z hand signs using a webcam.

🎯 Scope of the Project

The system detects and classifies static hand gestures representing A–Z alphabets.

The project focuses on single-hand gestures only.

Real-time video feed is processed through webcam.

Uses MediaPipe for hand landmark detection and ML model (Random Forest) for classification.

Output is displayed on the screen in real time.

Data collection, dataset creation, and model training scripts are included.

Note: This project does not cover full sentences or dynamic gestures; only static alphabet signs.

👤 Target Users

This project is useful for:

🧏 Hearing-impaired people

🎓 Students learning sign language

💻 Developers exploring computer vision / ML

🧪 Researchers working on gesture recognition

🏫 Educational institutions

🤝 Anyone wanting to understand basic sign language

⭐ High-Level Features

📸 Real-time Hand Tracking
Detects hand movement directly through webcam.

✋ 21 Hand Landmark Extraction
Uses MediaPipe to capture accurate finger & palm positions.

🧠 Machine Learning Classifier
Predicts A–Z alphabets using a trained Random Forest model.

💾 Custom Dataset Support
You can collect your own gesture images for training.

⚡ Fast and Efficient
Works smoothly on normal laptops without GPU.

🔍 Live Prediction Window
Shows detected hand sign in real-time.
