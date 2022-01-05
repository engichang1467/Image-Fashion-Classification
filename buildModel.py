import tensorflow as tf

# script to allow us to download MNIST dataset at load_data()
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# importFashion MNIST dataset
fashionMNIST = tf.keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashionMNIST.load_data()

trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Build our NN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(trainImages, trainLabels, epochs=10)

# Save our trained Model into a modle file
model.save('fashionReader.model')