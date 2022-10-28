# data
BATCH_SIZE = 8
VAL_DATA_PATH = './test'
TRAIN_DATA_PATH = './train'
RESIZE_SIZE = 224
TARGET_CLASSES = {
    'large_bowel': 1,
    'small_bowel': 2,
    'stomach': 3
}


# training
LR = 0.01
NUM_EPOCHS = 64

# model
INIT_MODEL_DEPTH = 32