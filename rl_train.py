import argparse
import torch
from config import *

def train():
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--algo", help="the rl algorithm to use", default=str choices=ALGOS, default="reinforce")
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")
  parser.add_argument("--batch_size", type=int, help="batch size", default=128)
  parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-3)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  args = parser.parse_args()

  train(args.train_data, args.output_dir, args.model_type, args.model_name, args.init_weight, args.batch_size, args.learning_rate, args.num_epochs, args.split_ratio)
