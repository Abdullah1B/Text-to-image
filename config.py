
BATCH_SIZE = 64 # number of seen smaples per iteration 

IMAGE_SIZE = 64 

IMAGE_DIR = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/" # point to the dir where the data is

CAPTION_DIR = "/kaggle/input/caption/caps.txt" # ponit where the captions for the image is locate it 
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU or cpu

EPOCHS = 500 # number of fowarwd and back 

NOISE_DIM = 100 

WORKERS = 2 # for the dataLoader 

LR_gen  = 0.0002 # learning rate for generator 
LR_disc = 0.0002 # learning rate for discriminator 


beta1 = 0.5

l1_coef = 50
l2_coef = 100











