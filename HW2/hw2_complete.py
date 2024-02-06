### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Input, layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Add, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import SeparableConv2D
##
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
def build_model1():
  model = Sequential([
    Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
  ])
  model.compile(optimizer=Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model
def build_model2():
  model = model = Sequential([
        # First Conv2D layer as specified
        Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),

        # Replacing subsequent Conv2D layers with depthwise separable convolutions
        SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False),
        BatchNormalization(),

        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

  model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  # Train the model
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0

  # No need to convert labels to one-hot encoding
  # train_labels, test_labels remain unchanged

  # Split the training set into training and validation sets
  val_split = int(len(train_images) * 0.8)
  val_images, val_labels = train_images[val_split:], train_labels[val_split:]
  train_images, train_labels = train_images[:val_split], train_labels[:val_split]

  ########################################
  ## Build and train model 1
  model1 = build_model1()

  # Make sure to compile the model with 'sparse_categorical_crossentropy'
  model1.compile(optimizer=Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

  history = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

  # Evaluate the model on the test set
  test_loss, test_accuracy = model1.evaluate(test_images, test_labels)

  # Plot training & validation accuracy values
  epochs_range = range(1, 51)
  plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
  plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')

  # Mark the test accuracy on the plot
  plt.scatter(len(epochs_range), test_accuracy, label='Test Accuracy', color='red')

  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(loc='upper left')
  plt.show()
  # compile and train model 1.
  # Path to your test image
  image_path = 'test_image_cat.png'

  # Load the image with the target size of 32x32 pixels
  image = load_img(image_path, target_size=(32, 32))

  # Convert the image to a numpy array and normalize it
  image = img_to_array(image) / 255.0

  # Add a batch dimension
  image = np.expand_dims(image, axis=0)
  # Make a prediction
  predictions = model1.predict(image)

  # Get the index of the highest probability
  predicted_class_index = np.argmax(predictions, axis=1)

  # CIFAR-10 classes
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Print the predicted class
  predicted_class_name = class_names[predicted_class_index[0]]
  print(f"Predicted class: {predicted_class_name}")
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
