import os
import json
import torch
from PIL import Image

class HatefulMemeDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,data_type, image_transform, tokenizer) -> None:
        super().__init__()
        self.data = [json.loads(l) for l in open(os.path.join(data_path,data_type+'.jsonl'))]
        self.data_dir = data_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load images on the fly.
        print(os.path.join(self.data_dir, self.data[index]["img"]))
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        text = self.data[index]["text"]
        label = self.data[index]["label"]
        
        return image, text, label

def collate_fn(self,batch):
    # Image Tensor
    tensor_img = torch.stack(
        [self.image_transform(row[0]) for row in batch]
    )

    # Tokenized Text Tensor 
    encoded_queries = self.tokenizer([row[1] for row in batch])
    lens = [len(row) for row in encoded_queries['input_ids']]
    text_tensor = torch.zeros(len(batch),max(lens))
    attention_mask = torch.zeros(len(batch),max(lens))
    
    for i_batch in range(len(batch)):
        length = lens[i_batch]
        text_tensor[i_batch, :length] = torch.Tensor(encoded_queries['input_ids'][i_batch])
        attention_mask[i_batch, :length] = torch.Tensor(encoded_queries['attention_mask'][i_batch])
    

    #Label Tensor
    label_tensor = torch.stack([torch.Tensor([row[2]]) for row in batch])

    return tensor_img,text_tensor,attention_mask,label_tensor