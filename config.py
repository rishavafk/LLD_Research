# config.py - global config
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths (adjust to your workspace)
EXDARK_IMG_ROOT = "/teamspace/studios/this_studio/ExDark/ExDark"
EXDARK_GT_ROOT  = "/teamspace/studios/this_studio/ExDark_Annotations/Groundtruth"
EXDARK_SPLIT_FILE = EXDARK_GT_ROOT + "/imageclasslist.txt"

CLASS_NAMES = [
    "Bicycle","Boat","Bottle","Bus","Car","Cat",
    "Chair","Cup","Dog","Motorbike","Person","Table"
]
NUM_CLASSES = len(CLASS_NAMES)

# training
IMG_SIZE = 640
BATCH = 8
EPOCHS = 5
NUM_WORKERS = 4
PIN_MEMORY = True

# model anchors + strides
STRIDES = [8,16,32]
ANCHORS = [
    [(10,13),(16,30),(33,23)],
    [(30,61),(62,45),(59,119)],
    [(116,90),(156,198),(373,326)]
]

# loss weights and thresholds
W_BOX = 1.0
W_OBJ = 1.0
W_CLS = 1.0
EPS = 1e-9
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

CONF_THRESH = 0.001
SCORE_THRESH = 0.25
NMS_IOU = 0.5
MAX_DETECTIONS = 300
