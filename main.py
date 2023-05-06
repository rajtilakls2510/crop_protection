import random
import numpy as np
import cv2
import time
from threading import Lock
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)

module1_data = pd.DataFrame({"time": [], "motion": [], "distance": []})
module2_data = pd.DataFrame({"time": [], "motion": [], "distance": []})

module_lock = Lock()

@app.route("/", methods=["GET"])
def sdf():
    return "HelloWorld"

@app.route("/module", methods=["POST"])
def module():
    global module1_data
    global module2_data
    module = request.form["module"]
    motion = request.form["motion"]
    distance = request.form["distance"]

    module_lock.acquire()
    if module == 1:
        module1_data = pd.concat([module1_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
    else:
        module2_data = pd.concat([module2_data, pd.DataFrame({"time": [time.time()], "motion": [motion], "distance": [distance]})])
    module_lock.release()
    return jsonify({"severity": random.randint(a=0, b=3)})

@app.route("/cam", methods=["POST"])
def cam():
    filestr = request.files['imageFile'].read()
    file_bytes = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    # Not yet decided what to do with image
    return "Received"

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)