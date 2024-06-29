import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
import numpy as np
import cv2

def load_model(weights_path, device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

def process_image(img_path, img_size=640):
    # Load image
    img = cv2.imread(img_path)
    img = letterbox(img, img_size, stride=32, auto=True)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dimension
    return img

def predict(model, img, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return pred

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = 'yolov5/runs/train/exp5/weights/best.pt'  # Update this path
model = load_model(weights, device)

# Process an image
image_path = 'testing_images/image2.jpg.jpg'  # Update this path
img = process_image(image_path)

# Prediction
pred = predict(model, img)
for i, det in enumerate(pred):  # detections per image
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape[2:]).round()

        # Print results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f'{model.names[c]} {conf:.2f}'
            print(label, xyxy)  # print class, confidence and bounding box coordinates
