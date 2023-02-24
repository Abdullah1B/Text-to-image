
BATCH_SIZE = 64 # number of seen smaples per iteration 
IMAGE_SIZE = 64 
IMAGE_DIR_TRAIN = "DATASET\\Images\\" # point to the dir where the data is
IMAGE_DIR_TEST = "DATASET\\Images\\" # point to the dir where the data is
CAPTION_DIR_TRAIN = "DATASET\\caption_v4.csv" # ponit where the captions for the image is locate it 
CAPTION_DIR_TEST = "DATASET\\caption_v4.csv" # ponit where the captions for the image is locate it 
DEVICE = 'cpu' # GPU or cpu
EPOCHS = 500 # number of fowarwd and back 
NOISE_DIM = 100 
TEXT_DIM  = 768
PROJECTED_SIZE = 128
FEATURES_NUM = 64
CHANNELS = 3
WORKERS = 2 # for the dataLoader 

LR_gen  = 0.0001 # learning rate for generator 
LR_disc = 0.0001 # learning rate for discriminator 
beta1 = 0.5
beta2 = 0.999
l1_coef = 70
l2_coef = 140











