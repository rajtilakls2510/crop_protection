import numpy as np
import cv2, io, base64
import time
from threading import Lock
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image

app = Flask(__name__)
cors = CORS(app)

module1_data = pd.DataFrame({"time": [], "motion": [], "distance": []})
module2_data = pd.DataFrame({"time": [], "motion": [], "distance": []})
cam_data = pd.DataFrame({"time": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "confidence": [], "class": []})
severities = [0,0]

cam_feed = None
mod1_feed = {"motion": 0, "distance": 0.0, "history": []}
mod2_feed = {"motion": 0, "distance": 0.0, "history": []}
mod1_override, mod2_override = False, False


module_lock = Lock()
cam_lock = Lock()

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

def alarm_detector():
    global module1_data, module2_data, cam_data

    # TODO: Process

    # Defining some of my own Hyperparameters
    window_size = 15
    motion_sensitivity = 3
    camera_sensitivity = 3

    # Record start and end timestamps for the current window
    start_time = module1_data['time'].iloc[0]
    end_time = start_time + pd.Timedelta(seconds=window_size)

    # Filter mod1 dataframe to only include rows within the current window
    mod1_window = module1_data.loc[(module1_data['time'] >= start_time) & (module1_data['time'] <= end_time)]

    # Check if motion is detected and note the distance
    mod1_motion_detected_flag = (mod1_window['motion'] == 1).sum() >= motion_sensitivity  # Require at least 3 motion detections in the window
    mod1_animal_distance = mod1_window['distance'].min() # Taking the minimum distance in the window

    # TODO: add time column in cam.csv
    # Filter camera dataframe to only include rows within the current window
    cam_window = cam_data.loc[(cam_data['time'] >= start_time) & (cam_data['time'] <= end_time)]
    # Turn the flag ON if animal is detected from the camera in the window
    animal_surity = cam_window.loc[(cam_window['confidence'] > 0.75) & (cam_window['class'] == 'animal')].shape[0]
    animal_visual_confirmation_flag = animal_surity >= camera_sensitivity

    # TODO: take confirmation and code below logic for severity
    # since we cannot detect no of animals, the severity here will be different.
    # if both module and camera detects animal and distance is in valid range, severity = 3 (highest)
    # if only 1 module and camera detects animal, severity = 2
    # if camera doest detect and module detects or vice versa, severity = 1
    # otherwise severity = 0

    # TODO: code to check if motion is detected and to calculate distance must be run in parallel for each module

    return 1, 1


@app.route("/live", methods=["GET"])
def live():
    img = Image.fromarray(cam_feed.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({"cam": {"img": img_base64} if cam_feed else None, "modules": [mod1_feed, mod2_feed]})


@app.route("/override", methods=["POST"])
def override():
    global mod1_override, mod2_override
    module = int(request.form["module"])
    override = bool(request.form["override"])
    if module == 1:
        mod1_override = override
    else:
        mod2_override = override
    return "Done"


@app.route("/control", methods=["POST"])
def control():
    global severities
    module = int(request.form["module"])
    control = bool(request.form["control"])
    if module == 1:
        severities[0] = control
    else:
        severities[1] = control
    return "Done"

@app.route("/module", methods=["POST"])
def module():
    global module1_data, module2_data, severities, mod1_feed, mod2_feed

    module = int(request.form["module"])
    motion = int(request.form["motion"])
    distance = float(request.form["distance"])

    module_lock.acquire()
    if module == 1:
        module1_data = pd.concat(
            [module1_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
        module1_data = module1_data[module1_data.time >= (time.time() - 100)]
        mod1_feed["motion"] = motion
        mod1_feed["distance"] = distance
    else:
        module2_data = pd.concat(
            [module2_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
        module2_data = module2_data[module2_data.time >= (time.time() - 100)]
        mod2_feed["motion"] = motion
        mod2_feed["distance"] = distance
    module1_data.to_csv("mod1.csv", index=False)  # TODO: Comment in production
    module2_data.to_csv("mod2.csv", index=False)  # TODO: Comment in production
    new_severities = alarm_detector()
    if module == 1 and not mod1_override and new_severities[0] > 0 and severities[0] == 0:
        mod1_feed["history"].append({"time": time.time(), "severity": new_severities[0]})
        severities[0] = new_severities[0]
    if module == 2 and not mod2_override and new_severities[1] > 0 and severities[1] == 0:
        mod2_feed["history"].append({"time": time.time(), "severity": new_severities[1]})
        severities[1] = new_severities[1]
    module_lock.release()
    return jsonify({"severity": severities[0] if module == 1 else severities[1]})


@app.route("/cam", methods=["POST"])
def cam():
    global cam_data, cam_feed
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
    animals = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    newpred = []
    for pred in results.pred[0]:
        if int(pred[-1]) in animals:
            newpred.append(pred)
    results.pred[0] = torch.stack(newpred) if len(newpred) > 0 else torch.Tensor([])
    results.render()
    cam_feed = results.ims[0]
    return "Received"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
