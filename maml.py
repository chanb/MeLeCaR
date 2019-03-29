import argparse
import os
import tensorflow as tf
import numpy as np

from config import *
from train import train
from utils.dataset_loader import read_from_pkl


# Meta-learn using MAML
def meta_train(training_dir, output_dir, model_type, model_name, task_batch_size, inner_learning_rate, meta_learning_rate, num_epochs):
  assert os.path.isdir(training_dir), "Training directory {} doesn't exist".format(training_dir)
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)

  if not os.path.isdir(output_dir):
    print("Creating directory {}".format(output_dir))
    os.makedirs(output_dir)

  workloads = [filename for filename in os.listdir(training_dir) if filename.endswith(DATASET_EXTENSION)]
  
  assert 1 <= task_batch_size <= len(workloads), "Specified task batch size is not within the range of [1, {}]".format(len(workloads))

  for epoch in range(num_epochs):
    for task in np.random.choice(workloads, task_batch_size, replace=False):
      inputs, outputs, input_dim, output_dim = read_from_pkl(os.path.join(training_dir, task))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--training_dir", type=str, help="the directory containing different tasks of training data", required=True)
  parser.add_argument("--output_dir", type=str, help="the saved model directory", required=True)
  parser.add_argument("--model_name", type=str, help="the saved model name", required=True)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")

  parser.add_argument("--task_batch_size", type=int, help="batch size of tasks", default=10)
  parser.add_argument("--inner_learning_rate", type=float, help="learning rate for inner loop", default=1e-3)
  parser.add_argument("--meta_learning_rate", type=float, help="learning rate for outer loop", default=1e-3)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument("--k_shot", type=int, help="number of datapoints to sample from for each task", default=5)

  args = parser.parse_args() 

  meta_train(args.training_dir, args.output_dir, args.model_type, args.model_name, args.task_batch_size, args.inner_learning_rate, args.meta_learning_rate, args.num_epochs)