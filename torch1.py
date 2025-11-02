import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import collections
import os

# ------------------ SETTINGS ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Number of last predictions to average (for stability)
SMOOTHING_WINDOW = 10  

# ------------------ LOAD MODEL ------------------
num_classes = 24  # Change if you have a different number of gestures
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load("resnet18_handsign.pth", map_location=device))
model.to(device)
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ------------------ CLASS LABELS ------------------
# Change these according to your dataset folders (alphabet order)
gesture_classes = sorted(os.listdir('gesture/Train'))
print("Loaded classes:", gesture_classes)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

pred_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI (optional) — focus on center or box area
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (128,128))

    # Preprocess
    tensor = transform(roi).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    pred_buffer.append(pred)
    # Majority vote for stability
    stable_pred = max(set(pred_buffer), key=pred_buffer.count)

    pred_label = gesture_classes[stable_pred]
    confidence = probs[0][stable_pred].item() * 100

    # Display
    cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Real-time Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
