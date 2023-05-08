import torch
import cv2
# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
animals = [15,16,17,18,19,20,21,22,23]
results = model(img)
print(results.pred)
newpred = []
for pred in results.pred[0]:
    if int(pred[-1]) in animals:
        newpred.append(pred)
print(newpred)

results.pred[0] = torch.stack(newpred) if len(newpred)>0 else torch.Tensor([])
results.render()
cv2.imshow("Image", results.ims[0])
cv2.waitKey(0)