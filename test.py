import argparse
import os
import tensorflow as tf


def test(test_data, model_dir):
  checkpoint_prefix = os.path.join(input_dir, "ckpt")
  root = tf.train.Checkpoint(optimizer=optimizer,
                            model=model,
                            optimizer_step=tf.train.get_or_create_global_step())

  root.restore(tf.train.latest_checkpoint(checkpoint_prefix))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_data", type=str, help="the pickle file containing the training input/output pairs", required=True)
  parser.add_argument("--model_dir", type=str, help="the checkpoint directory", required=True)

  args = parser.parse_args() 

  test(args.test_data, args.model_dir)