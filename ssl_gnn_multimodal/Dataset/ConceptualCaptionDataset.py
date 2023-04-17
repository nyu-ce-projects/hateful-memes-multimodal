import os
import json
import torch
from PIL import Image
from glob import glob
from pathlib import Path

class ConceptualCaptionDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,image_transform,tokenizer) -> None:
        super().__init__()
        self.data = glob(os.path.join(data_path,"*/*.jpg"), recursive = True)
        self.data_dir = data_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer     

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Load images on the fly.
        image = Image.open(self.data[index]).convert("RGB")
        text_file_path = self.data[index].replace("jpg","txt")
        text = Path(text_file_path).read_text().replace('\n','')
        
        return image, text
    
    def collate_fn(self,batch):
        # Image Tensor
        tensor_img = torch.stack(
            [self.image_transform(row[0]) for row in batch]
        )

        # Tokenized Text Tensor 
        encoded_queries = self.tokenizer([row[1] for row in batch])
        # print(encoded_queries)
        lens = [len(row) for row in encoded_queries['input_ids']]
        text_tensor = torch.zeros(len(batch),max(lens),dtype=torch.long)
        attention_mask = torch.zeros(len(batch),max(lens),dtype=torch.long)
        
        for i_batch in range(len(batch)):
            length = lens[i_batch]
            text_tensor[i_batch, :length] = torch.tensor(encoded_queries['input_ids'][i_batch])
            attention_mask[i_batch, :length] = torch.tensor(encoded_queries['attention_mask'][i_batch])
        

        return tensor_img,text_tensor,attention_mask