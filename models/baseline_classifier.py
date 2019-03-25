import tensorflow as tf

class Baseline(tf.keras.Model):
  def __init__(self, input_dim, output_dim):
    super(Baseline, self).__init__(name='')

    self.sequential = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(50, input_shape=input_dim),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(output_dim, input_shape=[None, 50]),
      tf.keras.layers.Activation("sigmoid")
    ])


  def call(self, x):
    x = self.sequential(x)
    return x


  def modify_input(self, input):
    return input[:, -1, :, :]