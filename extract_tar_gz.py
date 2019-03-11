import argparse
import os


EXTENSION = '.tar.gz'
BLKPARSE = '.blkparse'


def main(input_dir, output_dir):
  assert os.path.isdir(input_dir), "The input directory {} does not exist".format(input_dir)

  input_dir = input_dir.rstrip("/")
  if not os.path.isdir(output_dir):
    print("Creating output directory {}".format(output_dir))
    os.mkdir(output_dir)

  for file in os.listdir(input_dir):
    if not file.endswith(EXTENSION):
      continue
    
    print("Extracting and moving {}".format(file))
    execute_command("tar -xvzf {}".format(input_dir + '/' + file))
    execute_command("mv ./*{} {}".format(BLKPARSE, output_dir))
    

def execute_command(command):
  os.system(command)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str, help="the directory containing the tar.gz files", required=True)
  parser.add_argument("--output_dir", type=str, help="the directory containing content of the tar.gz files", required=True)
  args = parser.parse_args() 
  main(args.input_dir, args.output_dir)
