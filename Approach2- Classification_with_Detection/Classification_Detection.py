import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
import os

# Transformations for the detection model
detection_transform = transforms.Compose([
    transforms.ToTensor()
])

# Transformations for the classification model
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained Mask R-CNN model
detection_model = maskrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()

# Load pre-trained ResNet model and modify the final layer
classification_model = models.resnet18(pretrained=True)
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, 2)  # Adjust for two classes

# If you have a pre-trained classification model, load it
# classification_model.load_state_dict(torch.load('path_to_trained_classification_model.pth'))
classification_model.eval()

def detect_and_classify_cells(image_path, save_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image).unsqueeze(0)

    # Detect cells
    with torch.no_grad():
        detections = detection_model(image_tensor)
    
    # Extract bounding boxes and masks
    boxes = detections[0]['boxes'].cpu().numpy()
    masks = detections[0]['masks'].cpu().numpy()

    # Prepare the original image for annotation
    annotated_image = np.array(image)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Loop through detected cells
    for i, box in enumerate(boxes):
        # Extract the cell region using the bounding box
        x1, y1, x2, y2 = box.astype(int)
        cell_image = image.crop((x1, y1, x2, y2))

        # Transform the cell image for classification
        cell_tensor = classification_transform(cell_image).unsqueeze(0)

        # Classify the cell
        with torch.no_grad():
            output = classification_model(cell_tensor)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        # Map the label to a class name
        class_names = ['cholic', 'not_cholic']
        label_name = class_names[label]

        # Annotate the original image with the classification label
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    # Save the annotated image
    cv2.imwrite(save_path, annotated_image)
    print(f"Annotated image saved to {save_path}")

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"annotated_{filename}")
            detect_and_classify_cells(input_path, output_path)

# Set the input and output folders
input_folder = 'unknown'
output_folder = 'export'

# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)
