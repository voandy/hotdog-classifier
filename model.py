from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

from preprocessing import *


def build_model(width, height):
    # Define the model
    model = Sequential()

    # 2D convolution layers
    model.add(Conv2D(32, (3, 3), input_shape=(width, height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Downsizes images by 1/2 in this layer

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flattens layers from 3D to 1D further compressing features
    model.add(Flatten())

    # Regular densely connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout layer helps avoid over fitting by randomly setting 50% of the inputs to 0
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    return model


# Build and compile the model and define loss function
model = build_model(IMG_WIDTH, IMG_HEIGHT)
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Create TensorBoard logs
tensorboard = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True)

# Train the model with data from our generators
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=testing_samples // BATCH_SIZE,
    verbose=1,
    callbacks=[tensorboard]
)

# Save the model to disk
model.save('model_rmsprop.h5')
print("Model saved.")

# Print loss rate and accuracy
error_rate = model.evaluate_generator(validation_generator)
print("The model's loss rate is {0:0.2} (binary crossentropy) and accuracy is {0:.2%}"
      .format(error_rate[0], error_rate[1]))
