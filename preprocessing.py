import os.path

from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 100
IMG_HEIGHT = 100

EPOCHS = 100
BATCH_SIZE = 120

TRAIN_PATH = 'training-data/train'
TEST_PATH = 'training-data/test'

# Counts the number of training and testing samples in the directories
training_samples = sum([len(files) for r, d, files in os.walk(TRAIN_PATH)])
testing_samples = sum([len(files) for r, d, files in os.walk(TEST_PATH)])

# Augment images to prevent over-fitting and help the model identify true features
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
)

# Test image data only need to be rescaled to floats between 0 and 1
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Defines pipelines for training and testing data that loads batches of images, converts them to 3D numpy arrays
# and returns an iterator yielding the batches and their labels
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

validation_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)
