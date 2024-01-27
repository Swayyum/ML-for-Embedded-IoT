import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from person import person

rng = np.random.default_rng(2022)
list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]
# List comprehension to create a list of lengths of the names
name_lengths = [len(name) for name in list_of_names]

# Create a dictionary of person objects
people = {name: person(name, age, height) for name, age, height in zip(list_of_names, list_of_ages, list_of_heights_cm)}

for name in list_of_names:
  print("The name {:} is {:} letters long".format(name, len(name)))

# Convert lists of ages and heights into NumPy arrays
ages_array = np.array(list_of_ages)
heights_array = np.array(list_of_heights_cm)

# Calculate the average age
average_age = np.mean(ages_array)
print("Average Age:", average_age)

# Create a scatter plot of ages vs heights
plt.figure(figsize=(8, 6))
plt.scatter(ages_array, heights_array, color='blue', marker='o')
plt.title('Scatter Plot of Ages vs Heights')
plt.xlabel('Age')
plt.ylabel('Height in cm')
plt.grid(True)
plt.savefig('ages_heights_plot.png')  # Save the plot as a PNG file

########################################
# Here's the information for the second part, involving the linear
# classifier

# import the iris dataset as a pandas dataframe
iris_db = load_iris(as_frame=True) 
x_data = iris_db['data'] 
y_labels = iris_db['target'] # correct numeric labels
target_names = iris_db['target_names'] # string names

# Here's a starter example of plotting the data
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
l_colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(target_names):    
  plt.scatter(x_data[y_labels==n].iloc[:,0], 
              x_data[y_labels==n].iloc[:,1], 
              c=l_colors[n], label=target_names[n])
plt.xlabel(iris_db['feature_names'][0])
plt.ylabel(iris_db['feature_names'][1])
plt.grid(True)
plt.legend() # uses the 'label' argument passed to scatter()
plt.tight_layout()
# uncomment this line to show the figure, or use
# interactive mode -- plt.ion() --  in iPython
# plt.show()
plt.savefig('iris_data.png')


# Train a logistic regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg.fit(x_data, y_labels)

# Use the trained model's coefficients and intercept
W = log_reg.coef_
b = log_reg.intercept_
def classify_iris(x):
  # Initialize weights and biases
 # W = np.random.rand(3, 4)
  #b = np.random.rand(3)

  # Linear classification function
  y = np.argmax(np.matmul(W, x) + b)
  return y

# A function to measure the accuracy of a classifier and
# create a confusion matrix.  Keras and Scikit-learn have more sophisticated
# functions that do this, but this simple version will work for
# this assignment.
def evaluate_classifier(cls_func, x_data, labels, print_confusion_matrix=True):
  n_correct = 0
  n_total = x_data.shape[0]
  cm = np.zeros((3,3))
  for i in range(n_total):
    x = x_data[i,:]
    y = cls_func(x)
    y_true = labels[i]
    cm[y_true, y] += 1
    if y == y_true:
      n_correct += 1    
    acc = n_correct / n_total
  print(f"Accuracy = {n_correct} correct / {n_total} total = {100.0*acc:3.2f}%")
  if print_confusion_matrix:
    print(f"{12*' '}Estimated Labels")
    print(f"              {0:3.0f}  {1.0:3.0f}  {2.0:3.0f}")
    print(f"{12*' '} {15*'-'}")
    print(f"True    0 |   {cm[0,0]:3.0f}  {cm[0,1]:3.0f}  {cm[0,2]:3.0f} ")
    print(f"Labels: 1 |   {cm[1,0]:3.0f}  {cm[1,1]:3.0f}  {cm[1,2]:3.0f} ")
    print(f"        2 |   {cm[2,0]:3.0f}  {cm[2,1]:3.0f}  {cm[2,2]:3.0f} ")
    print(f"{40*'-'}")
  ## done printing confusion matrix  

  return acc, cm

## Now evaluate the classifier we've built.  This will evaluate the
# random classifier, which should have accuracy around 33%.
acc, cm = evaluate_classifier(classify_iris, x_data.to_numpy(), y_labels.to_numpy())


iris_db = load_iris(as_frame=True)
x_data = iris_db['data']
y_labels = iris_db['target']

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_labels_one_hot = encoder.fit_transform(y_labels.to_numpy().reshape(-1, 1))

  # Split the data into features (X) and labels (Y)
X = x_data.to_numpy()
Y = y_labels_one_hot

  # Create the TensorFlow/Keras model
tf_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),  # First hidden layer with 10 neurons
  tf.keras.layers.Dense(10, activation='relu'),  # Another hidden layer with 10 neurons
  tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons (one for each class)
])

  # Compile the model
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Train the model
tf_model.fit(X, Y, epochs=100)  # You can adjust the number of epochs

  # Evaluate the model on the same dataset
loss, accuracy = tf_model.evaluate(X, Y)
print(f"Model accuracy: {accuracy * 100:.2f}%")


np.argmax()
argmax on once axis

