import tensorflow as tf

class Baseline(tf.keras.Model):
  def __init__(self, input_dim, output_dim):
    super(Baseline, self).__init__(name='')

    self.sequential = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, input_shape=input_dim),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(128, input_shape=[None, 64]),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(64, input_shape=[None, 128]),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(output_dim, input_shape=[None, 64]),
      tf.keras.layers.Activation("sigmoid")
    ])


  def call(self, x):
    x = self._modify_input(x)
    x = self.sequential(x)
    return x


  def _modify_input(self, x):
    return x[:, -1, :, :]