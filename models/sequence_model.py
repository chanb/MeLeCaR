import tensorflow as tf

class SequenceModel(tf.keras.Model):
  def __init__(self):
    super(SequenceModel, self).__init__(name='')

    self.gru_1 = tf.keras.layers.GRU(hidden_units_1, input_shape=(5,))
    self.