import os
import json
import time
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import ResNet50_Weights


def store_image_detection_features(data_path):
    data = []
    device = 'cpu'
    BATCHSIZE = 2
    image_feature_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT
        ).to(device).eval()
    image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.ToTensor()
                ]
            )
    
    for data_type in ['train','dev','test']:
        data = [json.loads(l) for l in open(os.path.join(data_path,data_type+'.jsonl'))]
        for i in range(0,len(data),BATCHSIZE):
            image_path = [os.path.join(data_path, data[i]["img"]) for i in list(range(i,i+BATCHSIZE))]
            images = torch.stack([image_transform(Image.open(image_path[i]).convert("RGB")) for i in range(len(image_path))])

            start = time.perf_counter()
            outputs = image_feature_model(images)
            print(time.perf_counter()-start)
            for i,output in enumerate(outputs):

                if sum(output['scores']>0.4)>=5: #get features with more than 40% confidence score upto 20 features
                    indices = torch.argsort((output['scores']*(output['scores']>0.4)))[-20:]
                else:
                    indices = torch.argsort(output['scores'])[-5:] #get top 5 features
                
                masks = output['masks'][indices]
                img_feats = masks*images[i]
                print(img_feats.size())
                torch.save(img_feats,"{}.pt".format(os.path.splitext(image_path[i])[0]))

    
if __name__=="__main__":
    store_image_detection_features("./ssl_gnn_multimodal/Dataset/hateful_memes/")
    
