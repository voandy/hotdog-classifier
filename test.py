from predict import *

# Tests our model on the images in the test-images directory
for i in range(8):
    filename = 'test-images/test' + str(i) + '.jpg'

    print("The prediction for " + filename + " is:")
    predict_image(filename)
