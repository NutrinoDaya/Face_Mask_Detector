# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflowjs as tfjs
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import json


# initialize the initial learning rate,
#  number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
#The Directory of the files 
DIRECTORY = r"C:\Users\Administrator\Downloads\Compressed\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Grabbing The Images From The Files
print("[INFO] loading images...")
#Setting Data array to Store The Imagies In
data = []
labels = []
#Using A For Loop To Load imagies and Changing Them Into Arrays
# Then Adding Them To The Data Array
# + Adding The Label Of The Image To The Labels Array
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
# Using LabelBinarizer To Change The Label To Categorical Variables
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#Convert The Imagies/labels In The Data/labels Array To Numpy Arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
#Setting The Amount Of Imagies For Training And For Testing 
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
#We Use ImageDataGenerator to get Multiple Samples From The Same Image 
# This Way We Can Have More Samples That The Machine Can Learn From 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
#Creating A Basemodel
#imagenet Is A Pretrained Model For Imagies
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the the base model
#Adding Pooling to The Layers And Flattening Them 
#Adding Dense of 128 Neurons
#Dense(2) Represents The Number Of Layer That We Have 
# With_Mask / Without_Mask
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#Placing The FC Model Using The Input From The Net(Basemodel) and The Output
#This Will Be The Model That Will Be Trained
model = Model(inputs=baseModel.input, outputs=headModel)

#Looping Over The Layers In The Basemodel And Freeze Them 
# So That The Won't Update While The First Training Model Is Running 
for layer in baseModel.layers:
	layer.trainable = False

#Compiling Our Model  (Adam) Using .compile function 
#print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Training The Head OF The Network 
#Setting Steps Per Epoch
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Making Predection On The Testing Samples 
#Using .predict

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# For Each Image In The Testing Set We Need To Find The Index of the
# label(With_Mask,Without_Mask) with The largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

#Creating A Classifaction Report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Saving The Model To The Desk
print("[INFO] saving Face Mask model...")
model.save("mask_detector.model", save_format="h5")
tfjs.converters.save_keras_model(model, "tfjsmodel")

# Plotting The Training Loss And Accuracy 
#Then Saving It INto A PNG Image So That We Can Read The Results
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")