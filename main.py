"""
Real-Time Face Mask Detection Using OpenCV and Keras

This script performs real-time face mask detection using a pre-trained face detection model 
and a custom mask detection model. It uses OpenCV for face detection and video stream 
handling, and Keras for the mask detection model.

Functions:
---------
detect_and_predict_mask(frame, faceNet, maskNet)
    Detects faces in a frame, preprocesses them, and predicts whether they are wearing masks.

Variables:
---------
prototxtPath : str
    Path to the Caffe prototxt file for the face detection model.
weightsPath : str
    Path to the pre-trained weights for the face detection model.
faceNet : cv2.dnn_Net
    Pre-trained face detection model.
maskNet : keras.models.Model
    Pre-trained mask detection model.
vs : VideoStream
    Stream of frames captured from the webcam.

Usage:
-----
Run this script to start real-time face mask detection. Press 'q' to quit.
"""
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils
import time

def detect_and_predict_mask(frame, faceNet, maskNet):
    """
    Detect faces in a frame and predict mask usage for each detected face.

    Parameters
    ----------
    frame : np.ndarray
        A single frame captured from the video stream.
    faceNet : cv2.dnn_Net
        Pre-trained face detection model.
    maskNet : keras.models.Model
        Pre-trained mask detection model.

    Returns
    -------
    locs : list of tuple
        List of bounding box coordinates for detected faces [(startX, startY, endX, endY), ...].
    preds : list of np.ndarray
        List of mask prediction probabilities [(mask_prob, without_mask_prob), ...].
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Load face detection model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detection model
maskNet = load_model("Mel.model")

# Start video stream
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Main loop for real-time detection
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
