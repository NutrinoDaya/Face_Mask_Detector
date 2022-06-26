# Import Important Libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def DetectingMask_PredectingMask(frame, faceNet, maskNet):
	# Taking The Dimensions Of The Frame To Construct A Blobb From IT
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

# We Pass The Blob Through The Net To Obtain Face Detection
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	#Initializing An Array Of Faces , Locations , 
	# And The Predections Associated To It.
	
	faces = []
	locs = []
	preds = []

# Looping Over Detection
	for i in range(0, detections.shape[2]):
		# Extract The Confidence (The Probability) Associated With The
		# Detection
		confidence = detections[0, 0, i, 2]

		# Deleting Weak Detections If Their Confidence Is Less 
		# Than The minumum Confidence 
		if confidence > 0.5:
			# Compute The (x, y)-Coordinates Of 
			# The Bounding Box For The Object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Make Sure The Bounding Boxes Fall Withing The Dimensions 
			# Of The Frame  
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract The Face ROI, Convert It From BGR to RGB Channel
			# Ordering, Resize It To 224x224, And Preprocess It
			# Converting The faces From An Image To Array
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Appending The Faces And The Locations 
			# To Their Lists 
			faces.append(face)
			locs.append((startX, startY, endX, endY))

		# If At Least One Face Is Detected Make The Predection
	if len(faces) > 0:
		# For More Result We Will Make Batch Predections 
		#For All The Faces Detected At Once 
		#Not One By One
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Return A 2-Tuple Of The Face Locations And Their Corresponding
	# Locations
	return (locs, preds)

# Loading The Face Detector Model From The Desk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load The Mask Detector Model The We Created From The Desk
maskNet = load_model("mask_detector.model")

# Initialize The Live Video
print("[INFO] starting Live Video ...")
LiveVideo = VideoStream(src=0).start()

# Looping Over The Frames From The Live Video
while True:
	# Grab The Frame And Resize The Width To 1000 
	frame = LiveVideo.read()
	frame = imutils.resize(frame, width=1000)

	# Detect Faces In The Frame And Determine Wether They 
	#Wear A Mask Or Not
	(locs, preds) = DetectingMask_PredectingMask(frame, faceNet, maskNet)

	#Looping Over The Detected Faces Locations And Their Corresponding 
	#Locations
	for (box, pred) in zip(locs, preds):
		# Unpacking The Bounding Box And Predection
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Draw The Label Mask if (mask>withoutMask) 
		# Drawing The Color Of The Rectangle (RGB)
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Adding The Probability On Top Of The Face
		# We Use Max To Show The Higher Probability Only
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Drawing The Label And The Box On The Output Frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Showing The Output Frame 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#	If The User Presses q or Esc The Program Will Quit
	if key == ord("q"):
		break
	

# Cleaning Up
cv2.destroyAllWindows()
LiveVideo.stop()