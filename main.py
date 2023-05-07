import random
import numpy as np
import cv2
import time
from threading import Lock
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

app = Flask(__name__)
cors = CORS(app)

module1_data = pd.DataFrame({"time": [], "motion": [], "distance": []})
module2_data = pd.DataFrame({"time": [], "motion": [], "distance": []})
cam_data = pd.DataFrame({"time": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "confidence": [], "class": []})

module_lock = Lock()
cam_lock = Lock()

model = torch.hub.load("ultralytics/yolov5", "yolov5s")


@app.route("/", methods=["GET"])
def sdf():
    return "HelloWorld"


@app.route("/module", methods=["POST"])
def module():
    global module1_data
    global module2_data
    module = int(request.form["module"])
    motion = request.form["motion"]
    distance = request.form["distance"]

    module_lock.acquire()
    if module == 1:
        module1_data = pd.concat(
            [module1_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
        module1_data = module1_data[module1_data.time >= (time.time() - 100)]
    else:
        module2_data = pd.concat(
            [module2_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
        module2_data = module2_data[module2_data.time >= (time.time() - 100)]
    module1_data.to_csv("mod1.csv", index=False)  # TODO: Comment in production
    module2_data.to_csv("mod2.csv", index=False)  # TODO: Comment in production
    module_lock.release()
    return jsonify({"severity": random.randint(a=0, b=3)})


@app.route("/cam", methods=["POST"])
def cam():
    global cam_data
    filestr = request.files['imageFile'].read()
    file_bytes = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    detections = results.pandas().xyxyn[0]
    detections.replace(results.names, inplace=True)
    detections = detections[
        detections.name.isin(
            ["person", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"])].drop(
        columns="name") # TODO: Remove "person" in production
    detections["time"] = time.time()
    cam_lock.acquire()
    cam_data = pd.concat([cam_data, detections])
    cam_data = cam_data[cam_data.time >= (time.time() - 100)]
    cam_data.to_csv("cam.csv", index=False) # TODO: Comment in production
    cam_lock.release()
    return "Received"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
