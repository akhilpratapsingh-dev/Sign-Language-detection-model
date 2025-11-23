import pickle
import cv2
import mediapipe as mp
import numpy as np


# Load model
# Ensure the model file 'model.p' is in the same directory as this script.
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: The model file 'model.p' was not found. Please ensure it is in the correct directory.")
    exit()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# It's generally better to use static_image_mode=False for video streams.
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) 

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera. Check if another application is using it.")
    exit()

# Label mapping (Included for completeness, but not used in the final prediction logic)
# Your model outputs the character directly (e.g., 'A'), not an index (0).
labels_dict = {i: chr(65 + i) for i in range(26)}

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # --- FIX 1: PROCESS ONLY THE FIRST DETECTED HAND (42 Features) ---
    if results.multi_hand_landmarks:
        # Get the landmarks for the first hand detected (index 0)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 1. Draw Landmarks for the first hand
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # 2. Extract Absolute Coordinates (used for normalization and bounding box)
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        # 3. Calculate Relative Coordinates (Normalization)
        min_x = min(x_)
        min_y = min(y_)
        
        for landmark in hand_landmarks.landmark:
            # We subtract the min coordinates to make the hand position invariant
            data_aux.append(landmark.x - min_x)
            data_aux.append(landmark.y - min_y)
        
        # 4. Predict
        # The input data_aux is now a list of 42 normalized features.
        prediction = model.predict([np.asarray(data_aux)])
        
        # --- FIX 2: Handle String Output ---
        # The model returns the predicted character label directly (e.g., 'J', not 9).
        # We assign the prediction directly, avoiding the ValueError from int('J').
        predicted_char = str(prediction[0])

        # 5. Calculate and Display Bounding Box
        x1, y1 = int(min_x * W) - 10, int(min_y * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, predicted_char, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('Sign Language Detection', frame)
    # Press 'ESC' (key code 27) to exit the loop
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
