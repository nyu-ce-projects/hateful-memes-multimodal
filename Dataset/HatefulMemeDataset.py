import os
import json
import torch
from PIL import Image


class HatefulMemeDataset(torch.utils.data.Dataset):
    def __init__(self,data_path) -> None:
        super().__init__()
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load images on the fly.
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        text = self.data[index]["text"]
        label = self.data[index]["label"]
        
        return image, text, label