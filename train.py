import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import contrib

import config
from utils.parser_util import str2bool
from utils.dataset_loader import read_from_pkl
from models.baseline_classifier import Baseline


def step(model, loss_func, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss_func(model, inputs, targets)
    grads = tape.gradient(loss_value, model.trainable_variables)
  del tape
  return loss_value, grads


def train(train_data, output_dir, model_type, batch_size=10, learning_rate=1e-5, num_epochs=10, validation=False, split_ratio=0.7):
  assert model_type in config.MODEL_TYPES
  assert not validation or 0 <= split_ratio <= 1

  dataset, dataset_size, input_dim, output_dim = read_from_pkl(train_data)

  if validation:
    training_size = int(dataset_size * split_ratio)
    training_set = dataset.take(training_size).batch(batch_size)
    validation_set = dataset.skip(training_size).batch(dataset_size - training_size)
  else:
    training_set = dataset.batch(batch_size)

  if model_type == config.BASELINE:
    model = Baseline([None, input_dim], output_dim)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  global_step = tf.Variable(0)
  tfe = contrib.eager

  def loss(model, inputs, targets):
    logits = model(inputs)
    return tf.losses.sigmoid_cross_entropy(targets, logits)


  best_loss = None
  try:
    for epoch in range(num_epochs):
      print("EPOCH {} ==========================".format(epoch))
      tf.keras.backend.set_learning_phase(config.TRAIN)
      epoch_loss_avg = tfe.metrics.Mean()
      epoch_loss_avg.build()
      
      # Training Loop
      for inputs, targets in training_set:
        inputs = model.modify_input(inputs)
      
        # Optimize the model
        loss_value, grads = step(model, loss, inputs, targets)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

        # Track progress
        epoch_loss_avg.call(loss_value)

      # print(grads)
      print("Training Loss: {}".format(epoch_loss_avg.result()))

      # Perform validation
      if validation:
        tf.keras.backend.set_learning_phase(config.EVAL)
        for valid_inputs, valid_targets in validation_set:
          valid_inputs = model.modify_input(valid_inputs)
          valid_loss = loss(model, valid_inputs, valid_targets)

        print("Validation Loss: {}".format(valid_loss))
        if not best_loss or best_loss > valid_loss:
          best_loss = valid_loss
          print("Current best model. Saving model to {}...".format(output_dir))
          checkpoint_prefix = os.path.join(output_dir, "ckpt")
          root = tf.train.Checkpoint(optimizer=optimizer,
                                    model=model,
                                    optimizer_step=tf.train.get_or_create_global_step())

          root.save(checkpoint_prefix)
  except KeyboardInterrupt:
    print("Keyboard interrupt... Exiting training")


  if not validation:
    print("Saving model to {}...".format(output_dir))
    checkpoint_prefix = os.path.join(output_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer,
                              model=model,
                              optimizer_step=tf.train.get_or_create_global_step())

    root.save(checkpoint_prefix)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data", type=str, help="the pickle file containing the training data", required=True)
  parser.add_argument("--output_dir", type=str, help="the output checkpoint directory", required=True)
  parser.add_argument("--batch_size", type=int, help="batch size", default=128)
  parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.1)
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument("--model_type", type=str, choices=config.MODEL_TYPES, default="baseline", help="the model architecture to train")
  parser.add_argument("--validation", type=str2bool, help="whether or not to perform validation", default=False)
  parser.add_argument("--split_ratio", type=float, help="the training/validation split ratio", default=0.7)
  args = parser.parse_args()

  train(args.train_data, args.output_dir, args.model_type, args.batch_size, args.learning_rate, args.num_epochs, args.validation, args.split_ratio)

