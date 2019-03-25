import argparse
import numpy as np
import tensorflow as tf
from tensorflow import contrib

import config
from utils.dataset_loader import read_from_pkl
from models.baseline_classifier import Baseline

def step(model, loss_func, inputs, outputs):
  with tf.GradientTape() as tape:
    loss_value = loss_func(model, inputs, outputs)
    grads = tape.gradient(loss_value, model.trainable_variables)
  return loss_value, grads

def main(input_data, output_model, batch_size=10, learning_rate=1e-5, num_epochs=10):
  dataset = read_from_pkl(input_data).batch(batch_size)

  baseline = Baseline([None, 30], 10)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  global_step = tf.Variable(0)
  tfe = contrib.eager

  def loss(model, inputs, outputs):
    logits = model(inputs)
    return tf.losses.mean_squared_error(outputs, logits)

  train_loss_results = []

  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_loss_avg.build()
    
    # Training Loop
    for inputs, outputs in dataset:
      inputs = inputs[:,-1,:,:]
    
      # Optimize the model
      logits = baseline(inputs)
      loss_value, grads = step(baseline, loss, inputs, outputs)
      optimizer.apply_gradients(zip(grads, baseline.trainable_variables),
                                global_step)

      # Track progress
      epoch_loss_avg.call(loss_value)

    train_loss_results.append(epoch_loss_avg.result())    
    print("Epoch {}: Loss: {}".format(epoch, epoch_loss_avg.result()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_data", type=str, help="the pickle file containing the training input/output pairs", required=True)
  parser.add_argument("--output_model", type=str, help="the output model name", required=True)
  parser.add_argument("--batch_size", type=int, help="batch size")
  parser.add_argument("--learning_rate", type=float, help="learning rate")
  parser.add_argument("--num_epochs", type=int, help="number of epochs")
  args = parser.parse_args() 

  main(args.input_data, args.output_model, args.batch_size, args.learning_rate, args.num_epochs)

