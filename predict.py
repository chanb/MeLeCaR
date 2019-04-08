import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from config import *
from models.baseline_classifier import Baseline
from models.sequence_model import SequenceModel
from models.utils import get_full_model_name
from utils.dataset_loader import read_from_pkl

def get_block_trace(predict_raw_file, data_percentage):
    # predict_file = "webmail+online.cs.fiu.edu-110108-113008.16.blkparse"
    df = pd.read_csv(predict_raw_file, sep=' ',header = None)
    df.columns = ['timestamp','pid','pname','blockNo', \
                  'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']

    predictBlockTrace = df['blockNo'].tolist()
    predictBlockTrace = predictBlockTrace[:int(len(predictBlockTrace)*data_percentage)]
    return predictBlockTrace


def model_predict(train_data, trained_model_file_name, predict_data, batch_size=10):
    inputs, outputs, input_dim, output_dim = read_from_pkl(train_data)
    model = SequenceModel([None, input_dim], output_dim, 100)
    model.load_weights(trained_model_file_name)
    pred_result = model.predict(predict_data, batch_size)
    return pred_result

def Y_getBlockSeq(Y_pred_prob, eviction):
	x = []
	for i in range(len(Y_pred_prob[-1])):
		x.append(Y_pred_prob[-1][i])
	x = np.array(x)
	idx = np.argsort(x)
	idx = idx[:eviction]
	return idx

def predict(train_data, trained_model_file_name, predict_pkl_data, predict_raw_file, data_percentage, batch_size=10, cache_size=100):
	blocktrace = get_block_trace(predict_raw_file, data_percentage)
	# LFUDict = defaultdict(int)
	LRUQ = []
	hit = 0
	miss = 0
	loop_num = 0
	C = []
	evictCacheIndex = np.array([])
	for seq_number, block in enumerate(tqdm(blocktrace, desc="OPT")):
		# LFUDict[block] +=1
		loop_num += 1
		if block in C:
			hit+=1
			LRUQ.remove(block)
			LRUQ.append(block)
		else:
			evictPos = -1
			miss+=1
			# print("miss time and hit time")
			# print(miss)
			# print(hit)
			if len(C) == cache_size:
				if len(evictCacheIndex) == 0: # call eviction candidates
					pred_inputs, pred_outputs, pred_input_dim, pred_output_dim = read_from_pkl(predict_pkl_data)
					X_predict = pred_inputs
					Y_pred_prob = model_predict(train_data,trained_model_file_name,X_predict,batch_size)
                    # index of cache blocks that should be removed
					evictCacheIndex = Y_getBlockSeq(Y_pred_prob,int(0.7*cache_size))
                    #return Y_pred_prob, evictCacheIndex
                # evict from cache
				evictPos = evictCacheIndex[0]
				# print("test: print out the biggest probability position:")
				# print(Y_pred_prob[evictPos]) # test: print out the biggest probability position
				# print("evictPos:")
				# print(evictPos)
				# print("evictCacheIndex")
				# print(evictCacheIndex)
				# print("Y_pred_prob")
				# print(Y_pred_prob.shape)
				# print(Y_pred_prob[-1])
				# print("cache content")
				# print(C)
				# print("pred_inputs")
				# print(pred_inputs[0][0])
				evictBlock = C[evictPos]
				LRUQ.remove(evictBlock)
			if evictPos is -1:
				C.append(block)
			else:
				C[evictPos] = block
				evictCacheIndex = np.delete(evictCacheIndex, 0)
			LRUQ.append(block)
		if(loop_num%2000==0):
			print(hit / (hit + miss))
	hitrate = hit / (hit + miss)
	print(hitrate)
	return hitrate


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data", type=str, help="the pickle file containing the training data", required=True)
  parser.add_argument("--trained_model_file_name", type=str, help="the pickle file containing the trained model", required=True)
  parser.add_argument("--predict_pkl_data", type=str, help="pkl data to predict", required=True)
  parser.add_argument("--predict_raw_file", type=str, help="raw data to predict", required=True)
  parser.add_argument("--data_percentage", type=float, help="predict data percentage", required=True)
  parser.add_argument("--batch_size", type=int, help="batch size", default=10)
  parser.add_argument("--cache_size", type=int, help="learning rate", default=100)
  args = parser.parse_args()

  pred_hitrate = predict(args.train_data, args.trained_model_file_name, args.predict_pkl_data, args.predict_raw_file, args.data_percentage, args.batch_size, args.cache_size)
  # first test: hit rate = 0.03606070741700811
  # parameter:
  # python predict.py --train_data=webmail_16.pkl --trained_model_file_name=./train_output/webmail_16_train_output-gru.h5 --predict_pkl_data=online_09.pkl --predict_raw_file=online.cs.fiu.edu-110108-113008.9.blkparse --data_percentage=0.1 --batch_size=100 --cache_size=100