from Dataset.ConceptualCaptionDataset import ConceptualCaptionDataset
import torch_geometric.transforms as T
import torchvision
from transformers import AutoTokenizer,DistilBertTokenizer
from torch.utils.data import DataLoader
import math
import torch

def load_dataset():
    # Data
    print('==> Preparing data..')
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor()
        ]
    )
    model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = ConceptualCaptionDataset("../datasets/cc12m/",image_transform,tokenizer)
    n_dataset = len(dataset)
    n_dev = math.ceil(len(dataset)*0.05)
    n_test = math.ceil(len(dataset)*0.05)
    n_train = n_dataset - n_test - n_dev
    train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_dev, n_test))
    print(len(dev_dataset))
    print(train_dataset[0])
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,collate_fn=dataset.collate_fn)
    # x = next(iter(train_loader))


load_dataset()
