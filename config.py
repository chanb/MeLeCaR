import torch

BASELINE = "baseline"
GRU = "gru"
MODEL_TYPES = [BASELINE, GRU]


REINFORCE = "reinforce"
ALGOS = [REINFORCE]


CASA = "casa"
TASKS = [CASA]


ACCURACY = "accuracy"


DATASET_EXTENSION = ".pkl"


INPUTS = "inputs"
OUTPUTS = "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")