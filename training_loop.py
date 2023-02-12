import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import torchvision.transforms as transforms
import config as cfg
from torch.utils.data import DataLoader
from Text_image_dataset import Text_image_dataset as ds
from model import *

torch.manual_seed(0) 

training_dataset = ds(cap_dir=cfg.CAPTION_DIR_TRAIN,image_dir=cfg.IMAGE_DIR_TRAIN,
                      image_size=cfg.IMAGE_SIZE)

generator = Generator(channels=cfg.CHANNELS,noise_d=cfg.NOISE_DIM,text_dim=cfg.TEXT_DIM,
                      project_dim=cfg.PROJECTED_SIZE,
                      features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
generator.apply(weights_init)
discriminator = Discriminator(channels=cfg.CHANNELS,image_size=cfg.IMAGE_SIZE,
                              text_dim=cfg.TEXT_DIM,
                              project_dim=cfg.PROJECTED_SIZE,
                              features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
discriminator.apply(weights_init)

dataloader = DataLoader(training_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)

gen_opt = torch.optim.Adam(generator.parameters(),lr=cfg.LR_gen,betas=(cfg.beta1,cfg.beta2))
disc_opt = torch.optim.Adam(discriminator.parameters(),lr=cfg.LR_disc,betas=(cfg.beta1,cfg.beta2))

# loading test samples
test_data = ds(image_dir=cfg.IMAGE_DIR_TEST,cap_dir=cfg.CAPTION_DIR_TEST,image_size=cfg.IMAGE_SIZE)
test_loader = DataLoader(test_data,batch_size=cfg.BATCH_SIZE)
batch = next(iter(test_loader))
test_embeddings = batch['text_embedding'].to(cfg.DEVICE)
fixed_noise = torch.randn(test_embeddings.size(0),cfg.NOISE_DIM).to(cfg.DEVICE)
fixed_noise = fixed_noise.view(fixed_noise.size(0),cfg.NOISE_DIM,1,1)

BCE_loss = torch.nn.BCELoss()
l2_loss  = torch.nn.MSELoss ()
l1_loss  = torch.nn.L1Loss  ()

fake_writer = SummaryWriter('C:\\Users\\AJBas\\Desktop\\Project\\Text-to-image\\logs\\fake')
real_writer = SummaryWriter('C:\\Users\\AJBas\\Desktop\\Project\\Text-to-image\\logs\\real')
def train():
    step = 0

    for epoch in range(cfg.EPOCHS):

        for batch_idx, batch in enumerate(dataloader):

            real_image     = batch['right_images'].to(cfg.DEVICE)
            text_embedding = batch['text_embedding'].to(cfg.DEVICE)
            wrong_images   = batch['wrong_images'].to(cfg.DEVICE)
        
        # 1 for real and 0 for fake
            real_labels = torch.ones(real_image.size(0),device=cfg.DEVICE)
            fake_labels = torch.zeros(real_image.size(0),device=cfg.DEVICE)

            smoothed_real_labels = torch.FloatTensor(smooth_label(real_labels.cpu().numpy(), -0.1))
        
            real_labels = real_labels.to(cfg.DEVICE)
            smoothed_real_labels = smoothed_real_labels.to(cfg.DEVICE)
            fake_labels = fake_labels.to(cfg.DEVICE)

            discriminator.zero_grad()
        
            scores,real = discriminator(real_image,text_embedding)
            real_loss = BCE_loss(scores,smoothed_real_labels)

            scores,_ = discriminator(wrong_images,text_embedding)
            wrong_loss = BCE_loss(scores,fake_labels)

            noise = torch.randn(real_image.size(0),cfg.NOISE_DIM).to(cfg.DEVICE)
            noise = noise.view(noise.size(0),cfg.NOISE_DIM,1,1)
        
            fake_images = generator(text_embedding,noise)

            scores,_ = discriminator(fake_images,text_embedding)
            fake_loss = BCE_loss(scores,fake_labels)
            fake_score = scores

            d_loss = real_loss + fake_loss + wrong_loss
            d_loss.backward()
            disc_opt.step()
        # end of training the discriminator

        # training the generator

            generator.zero_grad()

            noise = torch.randn(real_image.size(0),cfg.NOISE_DIM)
            noise = noise.view(noise.size(0),cfg.NOISE_DIM,1,1)

            fake_images = generator(text_embedding,noise)

            scores, activation_fake = discriminator(fake_images,text_embedding)
            _ , activation_real     = discriminator(real_image,text_embedding)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)

            g_loss = BCE_loss(scores, real_labels) \
                         + cfg.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + cfg.l1_coef * l1_loss(fake_images, real_image)
            g_loss.backward()
            gen_opt.step()

            if batch_idx % 50 == 0:
                print(
                f"Epoch [{epoch}/{cfg.EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}, "
                )
                with torch.no_grad():
                    fake_images_test = generator(test_embeddings,fixed_noise)

                    img_real = torchvision.utils.make_grid(real_image[:32],normalize=True)
                    img_fake = torchvision.utils.make_grid(fake_images_test[:32],normalize=True)

                    real_writer.add_image('real',img_real,global_step=step)
                    fake_writer.add_image('Fake',img_fake,global_step=step)
                step += 1


train()

def generate(text):
    pass