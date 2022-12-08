import os
import json
import torch
from PIL import Image
import torchvision
from torchvision import transforms



class HatefulMemeDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,data_type) -> None:
        super().__init__()
        self.data = [json.loads(l) for l in open(os.path.join(data_path,data_type+'.jsonl'))]
        self.data_dir = data_path
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load images on the fly.
        print(os.path.join(self.data_dir, self.data[index]["img"]))
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        text = self.data[index]["text"]
        label = self.data[index]["label"]
        
        return image, text, label

     def fn(self,data):
        transforms.ToTensor()(image)
        tensor_img = torch.stack(
            [image_transform(image) for image in images]
        )