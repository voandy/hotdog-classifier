import sys

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

from preprocessing import *

# Squelch TensorFlow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
model = load_model('model_rmsprop.h5')
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)


# Prints a prediction
def print_prediction(prediction):
    if prediction < 0.5:
        print("The image is of a hotdog! Probability: {0:.2%}\n".format(1 - prediction))
    else:
        print("The image is not of a hotdog. Probability: {0:.2%}\n".format(prediction))


# Loads an image and makes a prediction using the model
def predict_image(filename):
    # Load image, downsize, scale and convert to array
    test_image = img_to_array(load_img(filename, target_size=(IMG_WIDTH, IMG_HEIGHT))) / 255.0

    # Expand array by 1 to match model
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)[0][0]
    print_prediction(prediction)


if len(sys.argv) > 1:
    predict_image(sys.argv[1])