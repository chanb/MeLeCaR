import torch

BASELINE = "baseline"
GRU = "gru"
MODEL_TYPES = [BASELINE, GRU]

ENV_LOCATION_PREFIX = "rl_envs/"

REINFORCE = "reinforce"
A2C = "a2c"
ALGOS = [REINFORCE, A2C]


HOME = "home"
WEBRESEARCH = "webresearch"
MAIL = "mail"
TASKS = [HOME, WEBRESEARCH, MAIL]
TASK_FILES = {HOME: "casa-110108-112108.{}.blkparse", WEBRESEARCH: "webresearch-030409-033109.{}.blkparse", MAIL: "cheetah.cs.fiu.edu-110108-113008.{}.blkparse"}
MAX_REQUESTS = [10000, 50000, 100000, 500000, 1000000, 2500000, 25000000]
CACHE_SIZE = [10, 30, 100, 1000, 10000]
FILE_INDEX = [3, 4, 5, 6]


ACCURACY = "accuracy"


DATASET_EXTENSION = ".pkl"


INPUTS = "inputs"
OUTPUTS = "outputs"

CUDA = 'cuda'
CPU = 'cpu'
MAP_LOCATION = CUDA if torch.cuda.is_available() else CPU
DEVICE = torch.device(MAP_LOCATION)
