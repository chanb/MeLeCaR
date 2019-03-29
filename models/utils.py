import tensorflow as tf
import math

def init_weight(in_dim, out_dim, name=None, stddev=1.0):
  return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=stddev/math.sqrt(float(in_dim))), name=name)


def init_bias(out_dim, name=None):
  return tf.Variable(tf.zeros([out_dim]), name=name)


def get_full_model_name(model_path, model_name):
  model_path = model_path.rstrip("/")
  return "{}/{}.h5".format(model_path, model_name)