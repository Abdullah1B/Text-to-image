import torch
from torchvision.models import inception_v3
import torchvision.transforems as tr
import scipy.linalg

from config import *


def load_inception_model(model_dir):
	
	inception_model = inception_v3(pretrained=False)
	inception_model.load_state_dict(torch.load(model_dir))
	inception_model.to(config.DEVICE)
	inception_model = inception_model.eval() # Evaluation mode
	inception_model.fc = torch.nn.Identity()

