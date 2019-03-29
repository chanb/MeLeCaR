import tensorflow as tf
from tensorflow import contrib

# tf.enable_eager_execution()
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

BASELINE = "baseline"
GRU = "gru"
MODEL_TYPES = [BASELINE, GRU]

ACCURACY = "accuracy"

DATASET_EXTENSION = ".pkl"