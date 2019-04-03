import argparse
import os
import tensorflow as tf
import numpy as np
import random

from config import *
from train import train
from utils.dataset_loader import read_from_pkl
from models.baseline_classifier import Baseline
from models.sequence_model import SequenceModel
from models.utils import get_full_model_name


# Meta-learn using MAML
def meta_train(training_dir, output_dir, model_type, model_name, task_batch_size, inner_learning_rate, meta_learning_rate, num_epochs, inner_epochs, k_shot):
  assert os.path.isdir(training_dir), "Training directory {} doesn't exist".format(training_dir)
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)

  if not os.path.isdir(output_dir):
    print("Creating directory {}".format(output_dir))
    os.makedirs(output_dir)

  workloads = [filename for filename in os.listdir(training_dir) if filename.endswith(DATASET_EXTENSION)]
  
  assert 1 <= task_batch_size <= len(workloads), "Specified task batch size is not within the range of [1, {}]".format(len(workloads))

  # Randomly initialize the first weights
  inputs, outputs, input_dim, output_dim = read_from_pkl(os.path.join(training_dir, workloads[0]))

  if model_type == BASELINE:
    model = Baseline([None, input_dim], output_dim)
  elif model_type == GRU:
    model = SequenceModel([None, input_dim], output_dim, 100)

  shape = inputs.shape
  model.build(shape)

  init_meta_weight = get_full_model_name(output_dir, "{}-meta_weight-0".format(model_name), model_type)

  print("Initialize weights to {}...".format(init_meta_weight))
  model.save_weights(init_meta_weight)
  meta_weights = None
  # Meta-learn loop
  for epoch in range(num_epochs):
    print("EPOCH {} =====================".format(epoch))
    # Task update
    curr_task_batch = np.random.choice(workloads, task_batch_size, replace=False)

    task_model_name = "{}-meta_weight-{}"
    init_meta_weight = get_full_model_name(output_dir, task_model_name.format(model_name, epoch), model_type)
    new_weights = []

    assert os.path.isfile(init_meta_weight), "{} weight file should exist".format(init_meta_weight)

    # Inner loop, sample batch of tasks
    for task in curr_task_batch:
      inputs, outputs, input_dim, output_dim = read_from_pkl(os.path.join(training_dir, task))

      # Sample k-shot
      k_samples = random.randint(0, len(inputs) - k_shot)

      # Recompile the model
      optimizer = tf.keras.optimizers.SGD(lr=inner_learning_rate)

      if model_type == BASELINE:
        model = Baseline([None, input_dim], output_dim)
      elif model_type == GRU:
        model = SequenceModel([None, input_dim], output_dim, 100)

      model.compile(optimizer, loss=tf.losses.sigmoid_cross_entropy, metrics=[ACCURACY])
      model.build(inputs.shape)
      model.load_weights(init_meta_weight)
      if meta_weights:
        print(meta_weights[0][0][:3])
        print(model.get_weights()[0][0][:3])

      # Train model for each task
      model.fit(inputs[k_samples:k_samples + k_shot], outputs[k_samples:k_samples + k_shot], batch_size=k_shot, epochs=inner_epochs, shuffle=False)

      if meta_weights:
        print(meta_weights[0][0][:3])
        print(model.get_weights()[0][0][:3])

      # Save new weights per task
      new_weight = get_full_model_name(output_dir, "{}-{}-{}".format(model_name, task, epoch), model_type)
      print("Saving task specific weights to {}...".format(new_weight))
      model.save_weights(new_weight)
      new_weights.append(model.get_weights())

    # Meta-train
    if model_type == BASELINE:
      model = Baseline([None, input_dim], output_dim)
    elif model_type == GRU:
      model = SequenceModel([None, input_dim], output_dim, 100)

    model.build(shape)
    model.load_weights(init_meta_weight)
    meta_weights = model.get_weights()

    # Compute meta update
    np_meta_weights = np.array(meta_weights)
    sum_meta_weights = -task_batch_size * np_meta_weights
    sum_new_weights = np.sum(new_weights, axis=0)

    meta_update_rate = meta_learning_rate / float(task_batch_size)
    out_1 = sum_new_weights + sum_meta_weights
    out_2 = out_1 * meta_update_rate
    np_meta_weights = np_meta_weights + out_2

    # Perform update
    for i, _ in enumerate(meta_weights):
      meta_weights[i] = np_meta_weights[i]
  
    model.set_weights(meta_weights)

    # Save new meta weights
    new_meta_weight = get_full_model_name(output_dir, "{}-meta_weight-{}".format(model_name, epoch + 1), model_type)
    print("Saving meta-weights to {}...".format(new_meta_weight))
    model.save_weights(new_meta_weight)

    print("")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--training_dir", type=str, help="the directory containing different tasks of training data", required=True)
  parser.add_argument("--output_dir", type=str, help="the saved model directory", required=True)
  parser.add_argument("--model_name", type=str, help="the saved model name", required=True)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="baseline", help="the model architecture to train")

  parser.add_argument("--task_batch_size", type=int, help="batch size of tasks", default=3)
  parser.add_argument("--inner_learning_rate", type=float, help="learning rate for inner loop", default=1e-3)
  parser.add_argument("--meta_learning_rate", type=float, help="learning rate for outer loop", default=1e-3)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument("--inner_epochs", type=int, help="number of epochs for task training", default=10)
  parser.add_argument("--k_shot", type=int, help="number of datapoints to sample from for each task", default=5)

  args = parser.parse_args() 

  meta_train(args.training_dir, args.output_dir, args.model_type, args.model_name, args.task_batch_size, args.inner_learning_rate, args.meta_learning_rate, args.num_epochs, args.inner_epochs, args.k_shot)