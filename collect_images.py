import os
import cv2

# Directory where data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create mapping for 26 letters (A–Z)
labels_dict = {i: chr(65 + i) for i in range(26)}
dataset_size = 100

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

for j in range(26):
    label = labels_dict[j]
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f'\nCollecting data for letter: {label}')
    print('Press "Q" when ready to start capturing...')

    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Ready to capture "{label}" - Press Q', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start capturing frames
    counter = 0
    print(f'Capturing {dataset_size} images for "{label}"...')
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(folder_path, f'{counter}.jpg'), frame)
        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

print("\nData collection complete for all letters.")
cap.release()
cv2.destroyAllWindows()