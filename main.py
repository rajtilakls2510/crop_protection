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

max_history = 10


cam_feed = cv2.imread("placeholder.png")
mod1_override, mod2_override = False, False
mod1_feed = {"module": 1, "motion": 1, "distance": 10.0, "severity": severities[0], "manual": mod1_override, "history": [{"time": time.time(), "severity": 1}, {"time": time.time(), "severity": 2}, {"time": time.time(), "severity": 3}]}
mod2_feed = {"module": 2, "motion": 0, "distance": 20.0, "severity": severities[1], "manual": mod2_override, "history": [{"time": time.time(), "severity": 1}, {"time": time.time(), "severity": 2}, {"time": time.time(), "severity": 3}]}


module_lock = Lock()
cam_lock = Lock()

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

def alarm_detector():
    global module1_data, module2_data, cam_data

    # TODO: Process
    # Defining some of my own Hyperparameters, Variables, and Flags
    sensor_window_size = 12
    cam_window_size = 5
    motion_sensitivity = 3
    domestic_animals = ["cat", "dog", "horse", "sheep", "cow", "person"]
    wild_animals = ["elephant", "bear", "zebra", "giraffe"]
    yolo_sensitivity = 0.35
    distance_sensitivity = 100
    module1_motion_flag = False
    module2_motion_flag = False

    try:
        # Fetch rows from dataframes within window size
        sorted_cam_df = cam_data.sort_values('time', ascending=False)
        t0_cam = time.time()
        cam_window = sorted_cam_df[(sorted_cam_df['time'] <= t0_cam) & (sorted_cam_df['time'] >= t0_cam-cam_window_size)]

        sorted_mod1_df = module1_data.sort_values('time', ascending=False)
        t0_mod1 = t0_cam
        module1_window = sorted_mod1_df[sorted_mod1_df['time'] >= (t0_mod1-sensor_window_size)]

        sorted_mod2_df = module2_data.sort_values('time', ascending=False)
        t0_mod2 = t0_cam
        module2_window = sorted_mod2_df[sorted_mod2_df['time'] >= (t0_mod2 - sensor_window_size)]

        # Module 1
        module1_motion_flag = (module1_window['motion'] == 1).sum() >= motion_sensitivity
        module1_distance = module1_window['distance'].min()

        # Module 2
        module2_motion_flag = (module2_window['motion'] == 1).sum() >= motion_sensitivity
        module2_distance = module2_window['distance'].min()

        # Camera
        domestic_rows = cam_window[cam_window['class'].isin(domestic_animals)]
        wild_rows = cam_window[cam_window['class'].isin(wild_animals)]
        domestic_rows = domestic_rows[domestic_rows['confidence'] > yolo_sensitivity]
        wild_rows = wild_rows[wild_rows['confidence'] > yolo_sensitivity]
        if not domestic_rows.empty:
            count_domestic_animals = domestic_rows.groupby(['time', 'class']).size().groupby("class").max().reset_index()[0].sum()
        else:
            count_domestic_animals = 0
        if not wild_rows.empty:
            count_wild_animals = wild_rows.groupby(['time', 'class']).size().groupby("class").max().reset_index()[0].sum()
        else:
            count_wild_animals = 0

        # Logic
        if count_domestic_animals == 0 and count_wild_animals == 0:
            severity = 0  # Level 0: No animals detected
        elif count_domestic_animals <= 5 and count_wild_animals < 1:
            severity = 1  # Level 1: Low alert, no wild animals
        elif 5 <= count_domestic_animals <= 15 or 1 <= count_wild_animals <= 5:
            severity = 2  # Level 2: Moderate alert, few wild or some domestic animals
        else:
            severity = 3  # Level 3: High alert, huge animals

        # Return logic
        if (module1_distance > distance_sensitivity):
            module1_motion_flag = False
        if (module2_distance > distance_sensitivity):
            module2_motion_flag = False
        if (module1_motion_flag or module1_distance <= distance_sensitivity) and (module2_motion_flag or module2_distance <= distance_sensitivity):
            return severity, severity
        elif (module1_motion_flag or module1_distance <= distance_sensitivity):
            return severity, 0
        elif (module2_motion_flag or module2_distance <= distance_sensitivity):
            return 0, severity
        else:
            return 0, 0

    except Exception as e:
        print("The error is: ", e)
        return severities

@app.route("/live", methods=["GET"])
def live():
    if cam_feed is not None:
        img = Image.fromarray(cam_feed.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read()).decode("ascii")

    return jsonify({"cam": {"img": img_base64} if cam_feed is not None else None, "modules": [mod1_feed, mod2_feed]})


@app.route("/override", methods=["POST"])
def override():
    global mod1_override, mod2_override
    module = int(request.form["module"])
    override = request.form["override"] == "true"
    if module == 1:
        mod1_override = override
        mod1_feed["manual"] = mod1_override
    else:
        mod2_override = override
        mod2_feed["manual"] = mod2_override
    return "Done"


@app.route("/control", methods=["POST"])
def control():
    global severities
    module = int(request.form["module"])
    control = int(request.form["control"])
    if module == 1:
        severities[0] = control
        mod1_feed["severity"] = severities[0]
    else:
        severities[1] = control
        mod2_feed["severity"] = severities[1]
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
    if module == 1 and not mod1_override:
        if new_severities[0] > 0 and severities[0] == 0:
            mod1_feed["history"].append({"time": time.time(), "severity": new_severities[0]})
            mod1_feed["history"] = mod1_feed["history"][-max_history:]
        severities[0] = new_severities[0]
        mod1_feed["severity"] = severities[0]
    if module == 2 and not mod2_override:
        if new_severities[1] > 0 and severities[1] == 0:
            mod2_feed["history"].append({"time": time.time(), "severity": new_severities[1]})
            mod2_feed["history"] = mod2_feed["history"][-max_history:]
        severities[1] = new_severities[1]
        mod2_feed["severity"] = severities[1]
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
