import torch
from torchvision.models import inception_v3
import scipy.linalg
import numpy as np

import config


def load_inception_model(model_dir):
	
	inception_model = inception_v3(pretrained=False)
	inception_model.load_state_dict(torch.load(model_dir))
	inception_model.to(config.DEVICE)
	inception_model = inception_model.eval() # Evaluation mode
	inception_model.fc = torch.nn.Identity()
	return inception_model


def matrix_sqrt(x):
    
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):

	return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(matrix_sqrt(np.matmul(sigma_x,sigma_y)))


def get_FID(dataLoader,gen_model,n_samples = 400,z_dim = 100):
	
	inception = load_inception_model()

	fake_list = []
	real_list = []

	current_samples = 0

	for batch in dataLoader:
		real_samples = batch['right_images'].to(config.DEVICE)
		real_samples = torch.nn.functional.interpolate(real_samples, size=( 299, 299), mode= "bilinear", align_corners=False)
		real_features = inception(real_samples).detach().to('cpu')
		real_list.append(real_features)

		noise = torch.randn(batch['right_images'].size(0),z_dim).to(config.DEVICE)
		noise = noise.view(noise.size(0),z_dim, 1, 1)

		fake_images = gen_model(batch["text_embedding"].to(config.DEVICE),noise)

		fake_samples  = torch.nn.functional.interpolate(fake_images, size=( 299, 299), mode= "bilinear", align_corners=False)
		fake_features = inception(fake_samples.to(config.DEVICE)).detach().to('cpu')
		fake_list.append(fake_features)

		current_samples += len(real_samples)
		if current_samples >= n_samples:
			break
	fake_list_all = torch.cat(fake_list)
	real_list_all = torch.cat(real_list)

	mu_fake = fake_list_all.mean(0)
	mu_real = real_list_all.mean(0)

	sigma_fake = torch.Tensor(np.cov(fake_list_all.detach().numpy(), rowvar= False))
	sigma_real = torch.Tensor(np.cov(real_list_all.detach().numpy(), rowvar= False))

	FID = 0
	with torch.no_grad():
		FID  = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()
	return FID