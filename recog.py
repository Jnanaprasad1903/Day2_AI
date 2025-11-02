import cv2
import torch
from torchvision import transforms, models
from PIL import Image

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL ------------------
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: ants & bees
model.load_state_dict(torch.load("resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# ------------------ CLASS NAMES ------------------
class_names = ['ants', 'bees']  # match your training folders

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ------------------ CAMERA SETUP ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not access webcam")
print("📸 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()

    # Display on frame
    display_text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, display_text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
