import tensorflow as tf
from models.utils import *

class Baseline(tf.keras.Model):
  def __init__(self, input_dim, output_dim):
    super(Baseline, self).__init__(name='')
    # self.flatten = tf.keras.layers.Flatten()
    # self.fully_connected_1 = tf.keras.layers.Dense(50, input_shape=input_dim, activation=tf.nn.relu)
    # # self.fully_connected_2 = tf.keras.layers.Dense(output_dim, input_shape=[None, 50], activation=tf.nn.sigmoid)
    # self.fully_connected_2 = tf.keras.layers.Dense(output_dim, input_shape=[None, 50])
    self.sequential = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, input_shape=input_dim),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(output_dim, input_shape=[None, 50]),
      tf.keras.layers.Activation("softmax")
    ])


  def call(self, x):
    x = self.sequential(x)
    return x
