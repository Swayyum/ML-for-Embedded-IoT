import sys
import numpy as np
# sys.path.append('../hw1')
# from hw1.hw1_complete import acc

import hw1_complete as hw
# from person import person as person

from sklearn.datasets import load_iris

def test_average_age():
  # test that the age numpy array was created and averaged
  assert hw.average_age == 38.0
  
def test_people_dict():
  # test that the people dictionary was constructed
  assert hw.people['Luisa'].age == 19

def test_person_string():
  # just check that the object for Elvis exists, and can be converted 
  # into a string that contains his age and height.
  s = str(hw.people['Elvis'])
  assert s.find('86') > 0
  assert s.find('182') > 0

# test that the classifier accepts a 4-D vector and returns an integer 0,1,or 2
def test_classifier():
  x_features = [1.0, 2.0, 3.0, 4.0]
  y = hw.classify_iris(x_features)
  assert y in [0,1,2]

# test that the classifier accuracy was measured and performed at at least random accuracy
def test_avg_low():
  assert hw.acc >= 0.3

# test that the classifier accuracy was measured and gave at least 65% accuracy
def test_avg_high():
  assert hw.acc >= 0.65

# test that the classifier accepts a 4-D vector and returns an integer 0,1,or 2
def test_tf_classifier_exists():
  x_features = np.array([[1.0, 2.0, 3.0, 4.0]])
  y = hw.tf_model(x_features)
  assert y.shape == (1,3)


def test_tf_classifier_performs():
  iris_db = load_iris(as_frame=True) 
  x_data = iris_db['data'] 
  y_labels = iris_db['target'] # correct numeric labels
  loss, acc  = hw.tf_model.evaluate(x_data, y_labels)
  print(f"TF Model: loss ={loss}, acc = {acc}")
  assert acc >= 0.5
  
