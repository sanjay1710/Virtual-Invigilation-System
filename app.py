<<<<<<< HEAD
from flask import Flask, render_template, flash, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
# from index import d_dtcn
import cv2
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'danger')

    return render_template('register.html')

@app.route('/logout/')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route("/home",methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
           return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")




@app.route('/upload_form')
def upload_form():
	return render_template('mcq.html')



import cv2
import numpy as np
import pyttsx3
import os
import datetime as dt
import matplotlib.pyplot as plt
from EAR_calculator import *
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import dlib
import time
import argparse
import cv2
from playsound import playsound
import os
import numpy as np
import pandas as pd
import pyttsx3
from matplotlib import style



@app.route("/start", methods=['GET', 'POST'])
def index():
    # Your start route code
    style.use('fivethirtyeight')

    
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Creating the dataset
    ear_list = []
    total_ear = []
    mar_list = []
    total_mar = []
    ts = []
    total_ts = []

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
    ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether raspberry pi camera shall be used or not")
    args = vars(ap.parse_args())

    # Declare a constant which will work as the threshold for EAR value, below which it will be regarded as a blink
    EAR_THRESHOLD = 0.3
    # Declare another constant to hold the consecutive number of frames to consider for a blink
    CONSECUTIVE_FRAMES = 20
    # Another constant which will work as a threshold for MAR value
    MAR_THRESHOLD = 14

    # Initialize two counters
    BLINK_COUNT = 0
    FRAME_COUNT = 0

    # Now, initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
    print("[INFO] Loading the predictor.....")
    detector = dlib.get_frontal_face_detector()
    args["shape_predictor"] = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Grab the indexes of the facial landmarks for the left and right eye respectively
    (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Now start the video stream and allow the camera to warm-up
    print("[INFO] Loading Camera.....")
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2)

    assure_path_exists("dataset/")
    count_sleep = 0
    count_yawn = 0

    # Load YOLO
    yolo_weights_path = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\yolov3.weights"
    yolo_config_path = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\yolov3.cfg"

    try:
        net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
        layer_names = net.getLayerNames()
        unconnected_layers = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
    except cv2.error as e:
        print("Error loading YOLO network:", e)

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Now, loop over all the frames and detect the faces
    while True:
        # Extract a frame
        frame = vs.read()
        cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        # Resize the frame
        frame = imutils.resize(frame, width=500)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        rects = detector(frame, 1)

        # Now loop over all the face detections and apply the predictor
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array
            shape = face_utils.shape_to_np(shape)

            # Draw a rectangle over the detected face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a number
            cv2.putText(frame, "Student", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            mouth = shape[mstart:mend]
            # Compute the EAR for both the eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            # live data write in CSV
            ear_list.append(EAR)

            ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # Compute the convex hull for both the eyes and then visualize it
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # Draw the contours
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

            MAR = mouth_aspect_ratio(mouth)
            mar_list.append(MAR / 10)

            # Object detection
            height, width, channels = frame.shape
            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 1, color, 3)

            # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place
            # Thus, count the number of frames for which the eye remains closed
            if EAR < EAR_THRESHOLD:
                FRAME_COUNT += 1

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                    count_sleep += 1
                    # Add the frame to the dataset as a proof of drowsy driving
                    cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                    engine.say(" Alert!")
                    engine.runAndWait()
                    cv2.putText(frame, " ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                    engine.say("Warning!")
                    engine.runAndWait()
                FRAME_COUNT = 0

            # Check if the person is yawning
            if MAR > MAR_THRESHOLD:
                count_yawn += 1
                cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
                cv2.putText(frame, " ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Add the frame to the dataset as a proof of drowsy driving
                cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
                engine.say(" Alert!")
                engine.runAndWait()

        # total data collection for plotting
        for i in ear_list:
            total_ear.append(i)
        for i in mar_list:
            total_mar.append(i)
        for i in ts:
            total_ts.append(i)
        # display the frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    a = total_ear
    b = total_mar
    c = total_ts

    df = pd.DataFrame({"EAR": a, "MAR": b, "TIME": c})
    df.to_csv("op_webcam.csv", index=False)
    df = pd.read_csv("op_webcam.csv")

    # Release the video stream and close all OpenCV windows
    cv2.destroyAllWindows()
    vs.stop()

    return render_template("index.html")

# Helper functions
# def eye_aspect_ratio(eye):
#     # Your EAR calculation code

# def mouth_aspect_ratio(mouth):
#     # Your MAR calculation code




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
=======
from flask import Flask, render_template, flash, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
# from index import d_dtcn
import cv2
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'danger')

    return render_template('register.html')

@app.route('/logout/')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route("/home",methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
           return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")




@app.route('/upload_form')
def upload_form():
	return render_template('mcq.html')



import cv2
import numpy as np
import pyttsx3
import os
import datetime as dt
import matplotlib.pyplot as plt
from EAR_calculator import *
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import dlib
import time
import argparse
import cv2
from playsound import playsound
import os
import numpy as np
import pandas as pd
import pyttsx3
from matplotlib import style



@app.route("/start", methods=['GET', 'POST'])
def index():
    # Your start route code
    style.use('fivethirtyeight')

    
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Creating the dataset
    ear_list = []
    total_ear = []
    mar_list = []
    total_mar = []
    ts = []
    total_ts = []

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
    ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether raspberry pi camera shall be used or not")
    args = vars(ap.parse_args())

    # Declare a constant which will work as the threshold for EAR value, below which it will be regarded as a blink
    EAR_THRESHOLD = 0.3
    # Declare another constant to hold the consecutive number of frames to consider for a blink
    CONSECUTIVE_FRAMES = 20
    # Another constant which will work as a threshold for MAR value
    MAR_THRESHOLD = 14

    # Initialize two counters
    BLINK_COUNT = 0
    FRAME_COUNT = 0

    # Now, initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
    print("[INFO] Loading the predictor.....")
    detector = dlib.get_frontal_face_detector()
    args["shape_predictor"] = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Grab the indexes of the facial landmarks for the left and right eye respectively
    (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Now start the video stream and allow the camera to warm-up
    print("[INFO] Loading Camera.....")
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2)

    assure_path_exists("dataset/")
    count_sleep = 0
    count_yawn = 0

    # Load YOLO
    yolo_weights_path = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\yolov3.weights"
    yolo_config_path = "D:\\Rudra 2023-24\\virtual assitance\\Automated\\yolov3.cfg"

    try:
        net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
        layer_names = net.getLayerNames()
        unconnected_layers = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
    except cv2.error as e:
        print("Error loading YOLO network:", e)

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Now, loop over all the frames and detect the faces
    while True:
        # Extract a frame
        frame = vs.read()
        cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        # Resize the frame
        frame = imutils.resize(frame, width=500)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        rects = detector(frame, 1)

        # Now loop over all the face detections and apply the predictor
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array
            shape = face_utils.shape_to_np(shape)

            # Draw a rectangle over the detected face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a number
            cv2.putText(frame, "Student", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            mouth = shape[mstart:mend]
            # Compute the EAR for both the eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            # live data write in CSV
            ear_list.append(EAR)

            ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # Compute the convex hull for both the eyes and then visualize it
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # Draw the contours
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

            MAR = mouth_aspect_ratio(mouth)
            mar_list.append(MAR / 10)

            # Object detection
            height, width, channels = frame.shape
            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 1, color, 3)

            # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place
            # Thus, count the number of frames for which the eye remains closed
            if EAR < EAR_THRESHOLD:
                FRAME_COUNT += 1

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                    count_sleep += 1
                    # Add the frame to the dataset as a proof of drowsy driving
                    cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                    engine.say(" Alert!")
                    engine.runAndWait()
                    cv2.putText(frame, " ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                    engine.say("Warning!")
                    engine.runAndWait()
                FRAME_COUNT = 0

            # Check if the person is yawning
            if MAR > MAR_THRESHOLD:
                count_yawn += 1
                cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
                cv2.putText(frame, " ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Add the frame to the dataset as a proof of drowsy driving
                cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
                engine.say(" Alert!")
                engine.runAndWait()

        # total data collection for plotting
        for i in ear_list:
            total_ear.append(i)
        for i in mar_list:
            total_mar.append(i)
        for i in ts:
            total_ts.append(i)
        # display the frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    a = total_ear
    b = total_mar
    c = total_ts

    df = pd.DataFrame({"EAR": a, "MAR": b, "TIME": c})
    df.to_csv("op_webcam.csv", index=False)
    df = pd.read_csv("op_webcam.csv")

    # Release the video stream and close all OpenCV windows
    cv2.destroyAllWindows()
    vs.stop()

    return render_template("index.html")

# Helper functions
# def eye_aspect_ratio(eye):
#     # Your EAR calculation code

# def mouth_aspect_ratio(mouth):
#     # Your MAR calculation code




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
>>>>>>> 059c5c6628fc9819c96f8fac1648dfeedf473faf
