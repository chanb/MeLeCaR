import argparse
import os
import tensorflow as tf

from config import *
from utils.dataset_loader import read_from_pkl
from models.baseline_classifier import Baseline
from models.utils import get_full_model_name


# Test the model
def test(test_data, model_dir, model_type, model_name):
  model_full_name = get_full_model_name(model_dir, model_name)
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)
  assert os.path.isfile(model_full_name), "{} does not exist".format(model_full_name)

  inputs, outputs, input_dim, output_dim = read_from_pkl(test_data)

  # Initialize model & optimizer
  print("Loading {} model from {}...".format(model_type, model_full_name))
  if model_type == BASELINE:
    model = Baseline([None, input_dim], output_dim)

  optimizer = tf.keras.optimizers.SGD()
  model.compile(optimizer, loss=tf.losses.sigmoid_cross_entropy, metrics=[ACCURACY])
  model.set_weights(model_full_name)

  model.evaluate(inputs, outputs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_data", type=str, help="the pickle file containing the training input/output pairs", required=True)
  parser.add_argument("--model_dir", type=str, help="the saved model directory", required=True)
  parser.add_argument("--model_name", type=str, help="the saved model name", required=True)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")

  args = parser.parse_args() 

  test(args.test_data, args.model_dir, args.model_type, args.model_name)