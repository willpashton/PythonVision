import tensorflow as tf
import matplotlib.pyplot
import PIL
import numpy as np
import pathlib
import sklearn
#os is used to validate paths
import os

from PIL import Image
from skimage import transform
from sklearn.utils import compute_class_weight
from tensorflow import keras, optimizers, losses
from keras import layers, activations
from keras import preprocessing
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

#directories

CAPTCHA_IMAGE_DIR = "./captchaimages/images/"
CAPTCHA_TEXT_DIR = "./captchatext/"
#print(os.path.isdir(CAPTCHA_IMAGE_DIR))
#print(os.path.isdir(CAPTCHA_TEXT_DIR))

#single image load

def loadSingleImage(path):
    img = keras.utils.load_img(path, target_size=(150,150))
    img = keras.utils.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img


#load datasets



imageSetGen = preprocessing.image.ImageDataGenerator(
    rescale = 1 / 255.0,
    zoom_range = 0.05,
    horizontal_flip = True,
    validation_split = 0.20
    )

imageSetTrain = imageSetGen.flow_from_directory(
    directory=CAPTCHA_IMAGE_DIR,
    target_size=(150,150),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=12
    )

imageSetValid = imageSetGen.flow_from_directory(
    directory=CAPTCHA_IMAGE_DIR,
    target_size=(150,150),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=12
    )

#model

# ROWS = 150
# COLS = 150
# shape = (ROWS, COLS, 3)

# model = applications.InceptionV3(weights='imagenet', include_top=False,input_shape=shape)
# for x in model.layers[0:150]:
#     x.trainable = False
# for x in model.layers[150::]:
#     x.trainable = True

# addModel = Sequential()
# addModel.add(model)
# addModel.add(GlobalAveragePooling2D())
# addModel.add(Dropout(0.5))
# addModel.add(Dense(12, activation="softmax")) #softmax used because sigmoid was awful

# endModel = addModel
# endModel.compile(loss="categorical_crossentropy",
#                  optimizer=optimizers.Adam(lr=1e-3), 
#                  metrics=["accuracy"])

#0.0001 (1e-4) was tried, as was 0.01

# tClasses = imageSetTrain.classes
# classWeights = compute_class_weight(
#     class_weight= "balanced",
#     classes= np.unique(tClasses),
#     y = tClasses
# )

# classWeights = dict(zip(np.unique(tClasses), classWeights))
# print(classWeights)

# callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath="./checkpoint/",
#     save_weights_only=False,
#     monitor="val_accuracy",
#     mode="max",
#     save_best_only=True,
#     verbose=1
# )

# history = endModel.fit(imageSetTrain, validation_data=imageSetValid, callbacks=callback, epochs = 5)

#save best model

bestModel = load_model("./bestmodel/bestimagemodel.h5", compile=False)
#bestModel.save("./bestmodel/bestimagemodel.h5", include_optimizer=False)

#create dict of labels

imageLabels = imageSetTrain.class_indices
imageLabels = dict((value, key) for key, value in imageLabels.items())
print(imageLabels)

img = loadSingleImage("./hyndranttest.png")

predict = bestModel.predict(img)
score = tf.nn.softmax(predict[0])
print(
     "This is most likely a {} with a {:.2f} percent confidence."
     .format(imageLabels[np.argmax(score)], 100 * np.argmax(score))
)

