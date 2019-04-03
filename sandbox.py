import tensorflow as tf
import numpy as np
from models.sequence_model import SequenceModel
from utils.dataset_loader import read_from_pkl


inputs, outputs, input_dim, output_dim = read_from_pkl("casa_6_01.pkl")
model = SequenceModel([None, input_dim], output_dim, 100)
model.build(inputs.shape)

test = model.get_weights()
print(test)
print(type(test))
print(len(test))

test2 = model.get_weights()

new_test = np.sum([test, test2], axis=0)
print(new_test)
print(type(new_test))

test3 = np.array(test) * 2
print(np.array(test)[0:1][:5])
print(test3[0:1][:5])

test4 = np.array(test) * 0.5