import argparse
import tensorflow as tf

from config import *
from train import train


def meta_train(training_dir, output_dir, model_type, model_name, batch_size, inner_learning_rate, meta_learning_rate, num_epochs):
  for epoch in num_epochs:
    pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--training_dir", type=str, help="the directory containing different tasks of training data", required=True)
  parser.add_argument("--output_dir", type=str, help="the saved model directory", required=True)
  parser.add_argument("--model_name", type=str, help="the saved model name", required=True)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")

  parser.add_argument("--batch_size", type=int, help="batch size of tasks", default=128)
  parser.add_argument("--inner_learning_rate", type=float, help="learning rate for inner loop", default=1e-3)
  parser.add_argument("--meta_learning_rate", type=float, help="learning rate for outer loop", default=1e-3)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument("--k_shot", type=int, help="number of datapoints to sample from for each task", default=5)

  args = parser.parse_args() 

  meta_train(args.training_dir, args.output_dir, args.model_type, args.model_name, args.batch_size, args.inner_learning_rate, args.meta_learning_rate, args.num_epochs)