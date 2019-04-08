import torch

BASELINE = "baseline"
GRU = "gru"
MODEL_TYPES = [BASELINE, GRU]


REINFORCE = "reinforce"
A2C = "a2c"
ALGOS = [REINFORCE, A2C]


HOME = "home"
WEBRESEARCH = "webresearch"
TASKS = [HOME, WEBRESEARCH]
TASK_FILES = {HOME: "casa-110108-112108.{}.blkparse", WEBRESEARCH: "webresearch-030409-033109.{}.blkparse"}
MAX_REQUESTS = [10000, 50000, 100000, 500000]
CACHE_SIZE = [10, 30, 100]
FILE_INDEX = [4, 5, 6]


ACCURACY = "accuracy"


DATASET_EXTENSION = ".pkl"


INPUTS = "inputs"
OUTPUTS = "outputs"

MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(MAP_LOCATION)