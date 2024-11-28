import torch
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from pathlib import Path

#Paths
model_path = './yolov5/best.pt'  
validation_images_dir = './yolov5/dataset/valid/images' 
validation_labels_dir = './yolov5/dataset/valid/labels' 


#Classes
class_names = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

#YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')

#Initialize the prediction lists
y_true = []
y_pred = []

#model.conf = 0.85 

#Iterate over validation images
image_paths = Path(validation_images_dir).glob("*.jpg")
for image_path in image_paths:
    #predictions
    results = model(str(image_path))
    pred_classes = results.pandas().xyxy[0]['name'].tolist()  #Predicted class

    #Maping the predicted class names to indices
    pred_indices = [class_names.index(cls) for cls in pred_classes if cls in class_names]

    #Load labels
    label_path = Path(validation_labels_dir) / (image_path.stem + ".txt")
    if label_path.exists():
        with open(label_path, 'r') as file:
            gt_classes = [int(line.split()[0]) for line in file.readlines()]  #Extract class ID
        #Append
        y_true.extend(gt_classes)
        y_pred.extend(pred_indices)

        # Log if no predictions are made
        if not pred_indices:
            print(f"No predictions for image: {image_path.name}")
    else:
        print(f"Warning: No ground truth found for {image_path.name}")

# Needed to adjust lengths due to missing predictions
if len(y_true) != len(y_pred):
    print(f"Adjusting lengths: y_true={len(y_true)}, y_pred={len(y_pred)}")
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

#Calculate confusion matrix and metrics
print("Calculating metrics...")
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
print("Confusion Matrix:")
print(cm)

#Save confusion matrix as a CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("confusion_matrix.csv")


#Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("\nClassification Report:")
print(report)

#Save classification report and predictions to CSV
predictions_df = pd.DataFrame({'Ground Truth': y_true, 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)
