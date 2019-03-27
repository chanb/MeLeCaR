import numpy as np
import tensorflow as tf
import pickle
import os

def read_from_pkl(filename, sequence_data=True):
  assert os.path.isfile(filename), "{} is not a file".format(filename)
  
  with open(filename, 'rb') as f:
    inputs, outputs, _ = pickle.load(f)

  assert inputs.shape[0] == outputs.shape[0]

  input_dim = tf.reshape(inputs[0][0], [-1]).shape[0] if sequence_data else tf.reshape(inputs[0], [-1]).shape[0]
  return tf.data.Dataset.from_tensor_slices((inputs, outputs)), inputs.shape[0], input_dim, tf.reshape(outputs[0], [-1]).shape[0]