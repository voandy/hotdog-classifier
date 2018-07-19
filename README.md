## Hot dog Image Classifier

This is a convolutional neural network trained from scratch to classify images of hot dogs with 82.45% accuracy.

![Comic](https://i.imgur.com/VBgGzbF.png)

Source: XKCD

Less than a decade ago the task of image classification was considered amongst the most difficult in computer science. We've come so far since then no because of considerable advances in hardware and processing power but because of great strides (pun intended) in software, namely convolutional neural networks.

In the field of machine learning, neural networks have been around for a long time but could never be used to efficiently train image classifiers. This was due to the exponentially multiplicative nature of neural networks compounded with the immense size of image files (3d arrays of size: width * length * channels). But the idea of combining neutral networks with kernel convolutions has made this task a very achievable one. In fact, libraries such as Keras and TensorFlow have made it easy even for the layman like myself.

This image classifier was built on the Keras API and trained on 1182 images of hot dogs and 2263 random images (half of which are of food so the model doesn't just learn to distinguish food from random) from the ImageNet database. It was trained twice once using the Adam optimisation algorithm and once with the RMSprop algorithm for 100 epochs each with RMSprop seeming to yield better results for the particular problem.

### Model Diagram

![Model](https://i.imgur.com/52NQxwt.png)

### Training

#### Loss

Blue: Adam / Red: RMSprop

![Loss](https://i.imgur.com/htR6ZdF.png)

#### Accuracy

Blue: Adam / Red: RMSprop

![Accuracy](https://i.imgur.com/asc2k24.png)

With this basic model I was able to obtain 82.45% accuracy on the testing data after around 30 minutes of training on a single GTX Titan Xp.

You can try using this model to make predictions on your own images by passing the filename as an argument to predict.py.