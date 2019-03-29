import tensorflow as tf

class SequenceModel(tf.keras.Model):
  def __init__(self, input_dim, output_dim, hidden_unit):
    super(SequenceModel, self).__init__(name='')
    self.sequential = tf.keras.Sequential()
    self.sequential.add(tf.keras.layers.GRU(hidden_unit, input_shape=input_dim))
    self.sequential.add(tf.keras.layers.Dense(100, input_shape=[None, hidden_unit]))
    self.sequential.add(tf.keras.layers.ReLU())
    self.sequential.add(tf.keras.layers.Dense(output_dim, input_shape=[None, 100]))
    self.sequential.add(tf.keras.layers.Activation("sigmoid"))
    

  def call(self, x):
    x = self._modify_input(x)
    return self.sequential(x)

  
  def _modify_input(self, x):
    curr_shape = x.shape
    x = tf.reshape(x, (tf.shape(x)[0], curr_shape[1], curr_shape[2] * curr_shape[3]))
    return x