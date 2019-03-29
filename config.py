import tensorflow as tf
from tensorflow import contrib

tf.enable_eager_execution()
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

BASELINE = "baseline"
MODEL_TYPES = [BASELINE]

ACCURACY = "accuracy"