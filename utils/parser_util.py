import argparse

from config import MAX_REQUESTS

def str2bool(value):
  if value.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif value.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def check_max_requests(value):
    ivalue = int(value)
    if ivalue not in MAX_REQUESTS:
      raise argparse.ArgumentTypeError("%s is an invalid max request value" % value)
    return ivalue