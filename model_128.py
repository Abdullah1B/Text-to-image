import torch
from torch import nn


def smooth_label(tensor, offset):
    return tensor + offset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Concat embedding


class concat_embed(nn.Module):
    def __init__(self, embedding_dim, project_dim):
        super(concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=project_dim),
            nn.BatchNorm1d(project_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)

        )

    def forward(self, image_features, text_embedding):

        projected_embedding = self.projection(text_embedding)
        replicated_embed = projected_embedding.repeat(
            4, 4, 1, 1).permute(2,  3, 0, 1)

        hidden_concat = torch.cat([image_features, replicated_embed], 1)

        return hidden_concat

# Generator Network


class Generator(nn.Module):
    def __init__(self, channels=3, noise_d=100,  text_dim=768, project_dim=128, features_num=64):
        super(Generator, self).__init__()
        self.channels      = channels
        self.noise_d       = noise_d
        self.text_dim      = text_dim
        self.project_dim   = project_dim
        self.features_num  = features_num

        self.text = nn.Sequential(
            nn.Linear(in_features=self.text_dim,out_features=self.project_dim),
            nn.BatchNorm1d(self.project_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.generator = nn.Sequential(
            self.Upsample_block(self.project_dim+self.noise_d,self.features_num * 16,kernel_size=4,stride=1,padding=0),
            self.Upsample_block(self.features_num * 16,self.features_num * 8,kernel_size=4,stride=2,padding=1),
            self.Upsample_block(self.features_num * 8 ,self.features_num  * 4,kernel_size=4,stride=2,padding=1),
            self.Upsample_block(self.features_num * 4 ,self.features_num  * 2,kernel_size=4,stride=2,padding=1),
            self.Upsample_block(self.features_num * 2 ,self.features_num     ,kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(in_channels=self.features_num,out_channels=self.channels,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )

    def Upsample_block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
    
    def forward(self, text_vec, noise):

        text = self.text(text_vec).unsqueeze(2).unsqueeze(3)
        combined = torch.cat([text, noise], 1)
        return self.generator(combined)


# Discriminator Network

class Discriminator(nn.Module):

    def __init__(self, channels=3, image_size=64, text_dim=768, project_dim=128, features_num=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.image_size = image_size
        self.text_dim = text_dim
        self.project_dim = project_dim
        self.features_num = features_num


        self.image_features = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,out_channels=self.features_num,kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.downsample_block(self.features_num,self.features_num *2, kernel_size=4,stride=2,padding=1),
            self.downsample_block(self.features_num * 2,self.features_num * 4, kernel_size=4,stride=2,padding=1),
            self.downsample_block(self.features_num * 4,self.features_num * 8, kernel_size=4,stride=2,padding=1),       
            self.downsample_block(self.features_num * 8,self.features_num * 16, kernel_size=4,stride=2,padding=1)       
            )
        self.concat_embeddings = concat_embed(self.text_dim, self.project_dim)

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=self.features_num * 16 + self.project_dim,out_channels=1,kernel_size=4,stride=1,padding=0,bias=False),
            nn.Sigmoid()
        )

    def downsample_block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
    
    def forward(self, image, text_embedding):
        image_features = self.image_features(image)

        combined = self.concat_embeddings(image_features, text_embedding)
        combined = self.discriminator(combined)

        return combined.view(-1, 1).squeeze(1), image_features
