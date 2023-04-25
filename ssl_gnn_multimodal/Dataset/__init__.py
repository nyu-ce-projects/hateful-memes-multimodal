from Dataset.ConceptualCaptionDataset import ConceptualCaptionDataset
from Dataset.HatefulMemeDataset import HatefulMemeDataset

import math
import torch
import torchvision
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,DistilBertTokenizer

def load_dataset(dataset_name,data_path,image_transform=None,tokenizer=None):
    print('==> Preparing data..')
    if image_transform is None:
        image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.ToTensor()
                ]
            )
    if tokenizer is None:
        # model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    if dataset_name=="HatefulMeme":
        train_dataset = HatefulMemeDataset(data_path,'train',image_transform,tokenizer)
        dev_dataset = HatefulMemeDataset(data_path,'dev',image_transform,tokenizer)
        test_dataset = HatefulMemeDataset(data_path,'test',image_transform,tokenizer)
        collate_fn = train_dataset.collate_fn
    elif dataset_name=="ConceptualCaption":
        dataset = ConceptualCaptionDataset(data_path,image_transform,tokenizer)
        n_dataset = len(dataset)
        n_dev = math.ceil(len(dataset)*0.05)
        n_test = math.ceil(len(dataset)*0.05)
        n_train = n_dataset - n_test - n_dev
        train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_dev, n_test))
        collate_fn = dataset.collate_fn
    else:
        raise NameError("No associated dataset found for the given name")
    
    return train_dataset,dev_dataset,test_dataset,collate_fn
