import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from resnet18 import ResNet, ResBlock

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=7) # 3 channel
model = ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=7) # grayscale
model.load_state_dict(torch.load('../model resnet 18 new.pth', map_location=device))
model.eval()

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_tensor = transforms.ToTensor()(roi_gray).unsqueeze_(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs.data, 1)
        predicted_emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][predicted[0]]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
