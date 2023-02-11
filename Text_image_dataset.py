import torch 
from torch import nn
from torchvision.io import read_image
from transformers import BertModel, BertTokenizer
import numpy as np
import torchvision.transforms as tr 
import pandas as pd



class Text_image_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir,cap_dir,image_size,transformer = None) -> None:

        self.image_dir      = image_dir
        self.cap_dir        = cap_dir
        self.transformer    = transformer
        self.image_size     = image_size 
        self.bert_model     = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        self.file_names,self.captions = self.get_file_names_caption()

        self.bert_model.eval()

    def __len__(self): 
        return len(self.file_names)

    def get_file_names_caption(self):
        df = pd.read_csv(self.cap_dir)
        
        file_names = df['file-name']
        captions = df['caption']
        
        return file_names, captions

    def get_text_embedding(self,text:str):
        
        max_len = 30

        tokenized_text = self.bert_tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = max_len, 
                            padding = 'max_length'
                        )

        input_ids = tokenized_text['input_ids']

        segment_ids = [1] * len(input_ids)

        tokens_tensor = torch.tensor([input_ids])
        segment_ids = torch.tensor([segment_ids])

        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor,segment_ids)
            hidden_states = outputs[2]

        token_vector = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vector, dim=0)


        return sentence_embedding


    def __getitem__(self,index):

        transform = tr.Compose([
                tr.CenterCrop(580),
                tr.Resize([self.image_size,self.image_size]),
            ])
        
        right_image = read_image(self.image_dir+self.file_names[index]) # tensor of the image 
        right_image = transform(right_image)
        right_image = (right_image - 127.5) / 127.5 # normlize the image from [0,255] to [-1,1]

        random_index = np.random.randint(0,self.__len__() - 1)
        if random_index == index:
            random_index = np.random.randint(0,self.__len__() - 1)

        wrong_image = read_image(self.image_dir+self.file_names[random_index])
        wrong_image = transform(wrong_image)
        wrong_image = (wrong_image - 127.5) / 127.5

        text_embedding = self.get_text_embedding(self.captions[index].replace(',',' '))

        sample = {

                  'right_images':right_image, 
                  'text_embedding':text_embedding,
                  'wrong_images':wrong_image

                  }

        return sample







