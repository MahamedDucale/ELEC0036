import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Create while loop to generate 30 Sigmoid weight files
i = 1
while i<=30:  
  # Load images and labels for training and test set
  train_images = mnist.train_images()
  train_labels = mnist.train_labels()
  test_images = mnist.test_images()
  test_labels = mnist.test_labels()

  # Normalize the images.
  train_images = (train_images / 255) - 0.5
  test_images = (test_images / 255) - 0.5

  # Flatten the images.
  train_images = train_images.reshape((-1, 784))
  test_images = test_images.reshape((-1, 784))

  # Build the model.
  model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])

  # Compile the model.
  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  # Train the model.
  model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
  )
  
  # Save weights as file 
  filename = "Sigmoid_files/Sigmoid_model_" + str(i) + ".h5"
  model.save_weights(filename)
  i += 1

  