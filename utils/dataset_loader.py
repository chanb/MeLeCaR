import numpy as np
import tensorflow as tf
import pickle
import os

def read_from_pkl(filename):
  assert os.path.isfile(filename), "{} is not a file".format(filename)
  
  with open(filename, 'rb') as f:
    inputs, outputs, _ = pickle.load(f)

  assert inputs.shape[0] == outputs.shape[0]

  return tf.data.Dataset.from_tensor_slices((inputs, outputs))