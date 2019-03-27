import tensorflow as tf

class SequenceModel(tf.keras.Model):
  def __init__(self, input_dim, output_dim, hidden_units):
    super(SequenceModel, self).__init__(name='')
    self.sequential = tf.keras.Sequential()
    prev_dim = input_dim
    for hidden_unit in hidden_units:
      self.sequential.add(tf.keras.layers.GRU(hidden_unit, input_shape=prev_dim))
      prev_dim = 