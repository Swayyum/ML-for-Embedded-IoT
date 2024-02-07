### Add lines to import modules as needed
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Add, AveragePooling2D, Dropout, GlobalAveragePooling2D, ReLU
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
    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(96, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(144, (3, 3), padding='same', activation='relu', use_bias=False)(x)  # Adjusted filter count
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = Flatten()(x)

    x = Dense(132, activation='relu')(x)  # Adjusted unit count
    x = BatchNormalization()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))

    # Adjusting initial Convolutional Layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First Block with adjusted filters and Skip Connection
    shortcut = x
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    shortcut = Conv2D(64, (1, 1), padding='same', activation='relu')(shortcut)  # Ensure matching dimensions
    x = Add()([x, shortcut])  # First skip connection

    # Second Block with increased filters and Skip Connection
    shortcut = x
    x = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    shortcut = Conv2D(128, (1, 1), padding='same', activation='relu', strides=(2, 2))(shortcut)  # Adjust for stride
    x = Add()([x, shortcut])  # Second skip connection

    # Third Block with further increased filters and Skip Connection
    shortcut = x
    x = Conv2D(128, (4, 4), padding='same', activation='relu', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    shortcut = Conv2D(128, (1, 1), padding='same', activation='relu', strides=(2, 2))(shortcut)  # Adjust for stride
    x = Add()([x, shortcut])  # Third skip connection

    # Global Average Pooling and an increased units Dense Layer before Output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Increased units
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='adjusted_model3')
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model50k():
  model = Sequential([
        Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),

        Dense(120, activation='relu'),
        Dense(10, activation='softmax')
    ])

  model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  def load_cifar10_data(file):
      with open(file, 'rb') as fo:
          cifar_dict = pickle.load(fo, encoding='bytes')
      return cifar_dict[b'data'], cifar_dict[b'labels']


  # Assuming you've extracted the CIFAR-10 dataset to 'cifar-10-batches-py' directory
  #cifar10_dir =  r'C:\Users\X390 Yoga\Desktop\Swayam\Intro to ML\cifar-10-python\cifar-10-batches-py'
  cifar10_dir = r'C:\Users\SirM\Desktop\Swayam\Intro to ML\cifar-10-batches-py'
  training_files = [os.path.join(cifar10_dir, 'data_batch_{}'.format(i)) for i in range(1, 6)]
  test_file = os.path.join(cifar10_dir, 'test_batch')

  train_images, train_labels = [], []
  for file in training_files:
      data, labels = load_cifar10_data(file)
      train_images.append(data)
      train_labels += labels

  train_images = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  train_labels = np.array(train_labels)

  test_data, test_labels = load_cifar10_data(test_file)
  test_images = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  test_labels = np.array(test_labels)

  # Normalize pixel values
  train_images, test_images = train_images / 255.0, test_images / 255.0

  ########################################
  ## Build and train model 1
  model1 = build_model1()

  # Make sure to compile the model with 'sparse_categorical_crossentropy'
  model1.compile(optimizer=Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

  history = model1.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

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
  # image_path = 'test_image_cat.png'
  #
  # # Load the image with the target size of 32x32 pixels
  # image = load_img(image_path, target_size=(32, 32))
  #
  # # Convert the image to a numpy array and normalize it
  # image = img_to_array(image) / 255.0
  #
  # # Add a batch dimension
  # image = np.expand_dims(image, axis=0)
  # # Make a prediction
  # predictions = model1.predict(image)
  #
  # # Get the index of the highest probability
  # predicted_class_index = np.argmax(predictions, axis=1)
  #
  # # CIFAR-10 classes
  # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  #
  # # Print the predicted class
  # predicted_class_name = class_names[predicted_class_index[0]]
  # print(f"Predicted class: {predicted_class_name}")
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  model2.summary()
  
  ### Repeat for model 3 and your best sub-50k params model
  model50k = build_model50k()
  model50k.summary()

  model3 = build_model3()
  model3.summary()
