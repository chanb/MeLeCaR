import tensorflow as tf
from tensorflow import contrib

tf.enable_eager_execution()
tfe = contrib.eager
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

BASELINE = "baseline"
MODEL_TYPES = [BASELINE]

EVAL = 0
TRAIN = 1