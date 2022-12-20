#%matplotlib inline
import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import timm
import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings
import timm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pandas_path  # Path style access for pandas
from tqdm import tqdm
import torch                    
from torch.backends import cudnn
import torchvision
import fasttext
import pandas_path
import albumentations as A
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from gpu_utils import print_gpu_util_every_sec
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer1 = SummaryWriter('comp_graph_cnn')
writer2 = SummaryWriter('comp_graph_vit')

parent_path = Path("/scratch/ps4364/HM")

data_dir = parent_path / "Data" / "hateful_memes"
img_tar_path = data_dir / "img.tar.gz"
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev_seen.jsonl"
test_path = data_dir / "test_seen.jsonl.jsonl"


dev_path_frame = pd.read_json(dev_path, lines=True)
dev_path_frame

train_path

train_samples_frame = pd.read_json(train_path, lines=True)
train_samples_frame

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def make_train_valid_dfs(path):
    dataframe = pd.read_json(path,lines=True)
    max_id = dataframe["id"].max() + 1 if not False else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

train,val=make_train_valid_dfs(train_path)

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions,labels, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.labels = list(labels)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        label = torch.Tensor([list(self.labels)]).long().squeeze()
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['label'] = self.labels[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

batch_size_multiplier = torch.cuda.device_count()

transforms=A.Compose([A.Resize(384, 384, always_apply=True),
                      A.Normalize(max_pixel_value=255.0, always_apply=True),])
dataset = CLIPDataset(train["img"].values,train["text"].values,train["label"].values,tokenizer=tokenizer,transforms=transforms,)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=32*batch_size_multiplier,num_workers=14,shuffle=True)

val_dataset = CLIPDataset(val["img"].values,val["text"].values,val["label"].values,tokenizer=tokenizer,transforms=transforms,)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=32*batch_size_multiplier,num_workers=14,shuffle=False)

image_path = "/scratch/ps4364/HM/Data/hateful_memes"
captions_path = "/scratch/ps4364/HM/Data/hateful_memes"
class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 384

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(self,language_module,vision_module,
                 language_feature_dim=300,vision_feature_dim=300,fusion_output_size=256,dropout_p=0.1,num_classes=1000):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.language_adder=nn.Linear(768,300)
        self.vision_module = vision_module
        self.vision_adder=nn.Linear(512,300)
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim), 
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=num_classes
        )
        
    def forward(self, x):
        language_op1=self.language_module(x['input_ids'].cuda(),x['attention_mask'])
        language_op = self.language_adder(language_op1)
        
        vision_op1=self.vision_module(x['image'])
        vision_op = self.vision_adder(vision_op1)
        combined = torch.cat(
            [language_op, vision_op], dim=1
        )
        fused = torch.nn.functional.relu(
            self.fusion(combined)
            )
        
        logits = self.fc(fused)
        #pred = torch.nn.functional.softmax(logits)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cnn = LanguageAndVisionConcat(language_module=TextEncoder(),vision_module=timm.create_model('resnet50',  True, num_classes=512)).to(device)
model_vit = LanguageAndVisionConcat(language_module=TextEncoder(),vision_module=timm.create_model('vit_base_patch16_384',  True,num_classes=512)).to(device)

model_cnn = nn.DataParallel(model_cnn)
model_vit = nn.DataParallel(model_vit)
cudnn.benchmark = True

print(model_cnn)

for batch in dataloader:
    break
batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}

writer1.add_graph(model_cnn, batch)
writer2.add_graph(model_vit, batch)
writer1.close()
writer2.close()

raise Exception()

from torchcontrib.optim import SWA
params_cnn = [
        {"params": model_cnn.module.vision_module.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model_cnn.module.language_module.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model_cnn.module.vision_adder.parameters(), model_cnn.module.language_adder.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
optimizer_cnn = torch.optim.AdamW(params_cnn, weight_decay=0.)

params_vit = [
        {"params": model_vit.module.vision_module.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model_vit.module.language_module.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model_vit.module.vision_adder.parameters(), model_vit.module.language_adder.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
optimizer_vit = torch.optim.AdamW(params_vit, weight_decay=0.)


scheduler_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cnn, mode="min", patience=CFG.patience, factor=CFG.factor
    )
scheduler_vit = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_vit, mode="min", patience=CFG.patience, factor=CFG.factor
    )


def loss_fn(x, y):
    x_noise=(torch.normal(10e-6, 10e-9, size=(x.size()))).to(device)
    x=x+x_noise
    x =  torch.nn.functional.normalize(x, dim=-1, p=2)
    y_noise=(torch.normal(10e-10, 10e-15, size=(y.size()))).to(device)
    y=y+y_noise
    y =  torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


import math
def ssl_train_model(train_loader,model_vit,optimizer_vit,scheduler_vit,model_cnn,optimizer_cnn,scheduler_cnn,num_epochs):
    #writer = SummaryWriter()
    phase = 'train'
    model_cnn.train()
    model_vit.train()
    f1_score_vit=0
    best_loss=math.inf
    for i in tqdm(range(num_epochs)):
        #tqdm_object = tqdm(train_loader, total=len(train_loader))
        total_loss=0
        for batch in train_loader:
            optimizer_cnn.zero_grad()
            optimizer_vit.zero_grad()
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            pred_vit = model_vit(batch)
            pred_cnn = model_cnn(batch)
            model_sim_loss=loss_fn(pred_cnn,pred_vit)
            loss = model_sim_loss.mean()
            loss.backward()
            optimizer_cnn.step()
            optimizer_vit.step()
            scheduler_cnn.step(loss)
            scheduler_vit.step(loss)
            total_loss+=loss.item()
        print('For -',i,'Loss:',total_loss)
        if total_loss<best_loss:
            best_loss=total_loss
            print("Saving!")
            torch.save(model_cnn,'./vitb16-r50-CNNPART-CASS-BERT-384-logits-v2-optimparams-100-nodp-agg-{}GPU.pt'.format(batch_size_multiplier))
            torch.save(model_vit,'./vitb16-r50-VITPART-CASS-BERT-384-logits-v2-optimparams-100-nodp-agg-{}GPU.pt'.format(batch_size_multiplier))
    
        #writer.add_scalar("Self-Supervised Loss/train", total_loss, i)
    #writer.flush()


#from torch.utils.tensorboard import SummaryWriter

print('Training CASS multimodally')
print_gpu_util_every_sec()
ssl_train_model(dataloader,model_vit,optimizer_vit,scheduler_vit,model_cnn,optimizer_cnn,scheduler_cnn,num_epochs=100)


