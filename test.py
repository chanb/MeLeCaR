import argparse
import os
import tensorflow as tf

import config


# Test the model
def test(test_data, model_dir, model_type):
  assert model_type in config.MODEL_TYPES
  assert os.path.isdir(model_dir)

  checkpoint_prefix = os.path.join(model_dir, "ckpt")
  root = tf.train.Checkpoint(optimizer=optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())

  root.restore(tf.train.latest_checkpoint(checkpoint_prefix))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_data", type=str, help="the pickle file containing the training input/output pairs", required=True)
  parser.add_argument("--model_dir", type=str, help="the checkpoint directory", required=True)
  parser.add_argument("--model_type", type=str, choices=config.MODEL_TYPES, default="baseline", help="the model architecture to train")

  args = parser.parse_args() 

  test(args.test_data, args.model_dir, args.model_type)