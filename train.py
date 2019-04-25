import argparse
import numpy as np
import os
import tensorflow as tf

from config import *
from utils.parser_util import str2bool
from utils.dataset_loader import read_from_pkl
from models.baseline_classifier import Baseline
from models.sequence_model import SequenceModel
from models.utils import get_full_model_name


# Train a model
def train(train_data, output_dir, model_type, model_name, init_weight=None, batch_size=10, learning_rate=1e-5, num_epochs=10, split_ratio=0.3):
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)
  assert 0 <= split_ratio < 1, "train/validation split needs to be within [0, 1)"
  assert not init_weight or os.path.isfile(init_weight), "the specified weight file does not exist: {}".format(init_weight)

  if not os.path.isdir(output_dir):
    print("Creating directory {}".format(output_dir))
    os.makedirs(output_dir)

  inputs, outputs, input_dim, output_dim = read_from_pkl(train_data)

  # Initialize model & optimizer
  if model_type == BASELINE:
    model = Baseline([None, input_dim], output_dim)
  elif model_type == GRU:
    model = SequenceModel([None, input_dim], output_dim, 100)

  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  model.compile(optimizer, loss=tf.losses.sigmoid_cross_entropy, metrics=[ACCURACY])

  if init_weight:
    model.build(inputs.shape)
    model.load_weights(init_weight)

  # Train model
  model_full_name = get_full_model_name(output_dir, model_name, model_type)
  
  try:
    print("Training model...")
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_full_name, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max', verbose=1)
    model.fit(inputs, outputs, batch_size=batch_size, epochs=num_epochs, validation_split=split_ratio, callbacks=[mcp_save], shuffle=False)
  except KeyboardInterrupt:
    print("Keyboard interrupt... Exiting training")

  # Save model
  if split_ratio == 0:
    print("Saving model to {}...".format(model_full_name))
    model.save_weights(model_full_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data", type=str, help="the pickle file containing the training data", required=True)
  parser.add_argument("--output_dir", type=str, help="the output model directory", required=True)
  parser.add_argument("--model_name", type=str, help="the output model name", required=True)
  parser.add_argument("--init_weight", type=str, help="a weight file that specifies how to initialize the weights of the model")
  parser.add_argument("--batch_size", type=int, help="batch size", default=128)
  parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-3)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")
  parser.add_argument("--split_ratio", type=float, help="the training/validation split ratio", default=0.3)
  args = parser.parse_args()

  train(args.train_data, args.output_dir, args.model_type, args.model_name, args.init_weight, args.batch_size, args.learning_rate, args.num_epochs, args.split_ratio)

