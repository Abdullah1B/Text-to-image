import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import torchvision.transforms as transforms
import config as cfg
from torch.utils.data import DataLoader
from Text_image_dataset import Text_image_dataset as ds
from model import *
from model_128 import Generator as Gen_128 , Discriminator as Disc_128

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


torch.manual_seed(0) 

class training_loop():
    def __init__(self,use_128 = False) -> None:

        self.training_dataset = ds(cap_dir=cfg.CAPTION_DIR_TRAIN,image_dir=cfg.IMAGE_DIR_TRAIN,
                            image_size=cfg.IMAGE_SIZE)
        if use_128:
            self.generator = Gen_128(channels=cfg.CHANNELS,noise_d=cfg.NOISE_DIM,text_dim=cfg.TEXT_DIM,
                            project_dim=cfg.PROJECTED_SIZE,
                            features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
            self.generator.apply(weights_init)

            self.discriminator = Disc_128(channels=cfg.CHANNELS,image_size=cfg.IMAGE_SIZE,
                                    text_dim=cfg.TEXT_DIM,
                                    project_dim=cfg.PROJECTED_SIZE,
                                    features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
            self.discriminator.apply(weights_init)
        else:
            self.generator = Generator(channels=cfg.CHANNELS,noise_d=cfg.NOISE_DIM,text_dim=cfg.TEXT_DIM,
                            project_dim=cfg.PROJECTED_SIZE,
                            features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
            self.generator.apply(weights_init)
            self.discriminator = Discriminator(channels=cfg.CHANNELS,image_size=cfg.IMAGE_SIZE,
                                    text_dim=cfg.TEXT_DIM,
                                    project_dim=cfg.PROJECTED_SIZE,
                                    features_num=cfg.FEATURES_NUM).to(cfg.DEVICE)
            self.discriminator.apply(weights_init)

        self.dataloader = DataLoader(self.training_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)

        self.gen_opt = torch.optim.Adam(self.generator.parameters(),lr=cfg.LR_gen,betas=(cfg.beta1,cfg.beta2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(),lr=cfg.LR_disc,betas=(cfg.beta1,cfg.beta2))

        # loading test samples
        self.test_data = ds(image_dir=cfg.IMAGE_DIR_TEST,cap_dir=cfg.CAPTION_DIR_TEST,image_size=cfg.IMAGE_SIZE)
        self.test_loader = DataLoader(self.test_data,batch_size=cfg.BATCH_SIZE)
        self.batch = next(iter(self.test_loader))
        self.test_embeddings = self.batch['text_embedding'].to(cfg.DEVICE)
        self.fixed_noise = torch.randn(self.test_embeddings.size(0),cfg.NOISE_DIM).to(cfg.DEVICE)
        self.fixed_noise = self.fixed_noise.view(self.fixed_noise.size(0),cfg.NOISE_DIM,1,1)

        self.BCE_loss = torch.nn.BCELoss()
        self.l2_loss  = torch.nn.MSELoss ()
        self.l1_loss  = torch.nn.L1Loss  ()

        self.fake_writer = SummaryWriter('C:\\Users\\AJBas\\Desktop\\Project\\Text-to-image\\logs\\fake')
        self.real_writer = SummaryWriter('C:\\Users\\AJBas\\Desktop\\Project\\Text-to-image\\logs\\real')
    def train(self,train_on_notebook = False):
        step = 0

        for epoch in range(cfg.EPOCHS):

            for batch_idx, batch in enumerate(self.dataloader):

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

                self.discriminator.zero_grad()
            
                scores,real = self.discriminator(real_image,text_embedding)
                real_loss = self.BCE_loss(scores,smoothed_real_labels)

                scores,_ = self.discriminator(wrong_images,text_embedding)
                wrong_loss = self.BCE_loss(scores,fake_labels)

                noise = torch.randn(real_image.size(0),cfg.NOISE_DIM).to(cfg.DEVICE)
                noise = noise.view(noise.size(0),cfg.NOISE_DIM,1,1)
            
                fake_images = self.generator(text_embedding,noise)

                scores,_ = self.discriminator(fake_images,text_embedding)
                fake_loss = self.BCE_loss(scores,fake_labels)
                fake_score = scores

                d_loss = real_loss + fake_loss + wrong_loss
                d_loss.backward()
                self.disc_opt.step()
            # end of training the discriminator

            # training the generator

                self.generator.zero_grad()

                noise = torch.randn(real_image.size(0),cfg.NOISE_DIM).to(cfg.DEVICE)
                noise = noise.view(noise.size(0),cfg.NOISE_DIM,1,1)

                fake_images = self.generator(text_embedding,noise)

                scores, activation_fake = self.discriminator(fake_images,text_embedding)
                _ , activation_real     = self.discriminator(real_image,text_embedding)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                g_loss = self.BCE_loss(scores, real_labels) \
                            + cfg.l2_coef * self.l2_loss(activation_fake, activation_real.detach()) \
                            + cfg.l1_coef * self.l1_loss(fake_images, real_image)
                g_loss.backward()
                self.gen_opt.step()

                if batch_idx % 50 == 0:
                    print(
                    f"Epoch [{epoch}/{cfg.EPOCHS}] Batch {batch_idx}/{len(self.dataloader)} \
                    Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}, "
                    )
                    with torch.no_grad():
                        fake_images_test = self.generator(self.test_embeddings,self.fixed_noise)

                        img_real = torchvision.utils.make_grid(real_image[:32],normalize=True)
                        img_fake = torchvision.utils.make_grid(fake_images_test[:32],normalize=True)
                        if train_on_notebook:
                            self.show_tensor_images(fake_images_test)
                        self.real_writer.add_image('real',img_real,global_step=step)
                        self.fake_writer.add_image('Fake',img_fake,global_step=step)
                    step += 1


    def show_tensor_images(self,image_tensor, num_images=25, size=(3, 64, 64), fig_size = 8):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        plt.figure(figsize=[fig_size, fig_size])

        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid   = make_grid(image_unflat[:num_images], nrow=5)
        
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()



    def generate(self,text,path_model = "path/pre_trained",use_pre_train = False,use_128 = False):
        if use_pre_train:
            if use_128:
                gen = Gen_128()
                gen.load_state_dict(path_model)
            else:
                gen = Generator()
                gen.load_state_dict(path_model)
                
        gen = self.generator
        gen.eval()
        text = text.replace(',',' ')
        text_embedding = self.training_dataset.get_text_embedding(text)
        text_embedding = text_embedding.view(1,-1).to(cfg.DEVICE)


        noise = torch.randn(1,cfg.NOISE_DIM)
        noise = noise.view(noise.size(0),cfg.NOISE_DIM,1,1).to(cfg.DEVICE)

        with torch.no_grad():
            fake_image = gen(text_embedding,noise)

        self.show_tensor_images(fake_image,fig_size=6)
        gen.train()
        return fake_image 
