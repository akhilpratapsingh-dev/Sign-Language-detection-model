import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

# Loop through A–Z folders
for label in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        continue
    print(f'Processing folder: {label}')
    
    for img_path in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, data_aux = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(label)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created successfully.") 