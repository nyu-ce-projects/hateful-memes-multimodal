import torch
import torch.nn as nn
import clip
from torchvision import models,transforms
from torch.autograd import Variable
from Dataset.HatefulMemeDataset import HatefulMemeDataset


def main(device):
    model, preprocess = clip.load("ViT-B/32", device)
    model = model.to(device)
    dataset = HatefulMemeDataset('./data','train') 
    print(dataset[0])   
    # image_some_model()
    # bert()
    # create_graph()

    # GCN()
    device = 'cpu'
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    
    imgTensor = transforms.ToTensor()(dataset[1][0]).unsqueeze(0).to(device)
    # test_transforms = transforms.Compose([transforms.Resize(224),
    #                                   transforms.CenterCrop(224),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                       std=[0.229, 0.224, 0.225])
    #                                  ])

    # image_tensor = test_transforms(dataset[0][0]).float()
    # image_tensor = image_tensor.unsqueeze_(0)

    # input = Variable(image_tensor)
    output = model(imgTensor)[0]
    print(output['masks'].shape)

    # for intMask in range(output['masks'].shape[0]):
    #     if output['scores'][intMask].item()>=0.7:



    # image_input = preprocess(dataset[0][0])  #.unsqueeze(0).to(device)
    # text_inputs = clip.tokenize(dataset[0][1]).to(device)
    # print(image_input.shape)
    # # with torch.no_grad():
    # #     image_features = model.encode_image(image_input)
    # #     text_features = model.encode_text(text_inputs)
    # # print(image_features)
    # print("TextAkash")
    # print(text_inputs.shape)
    # print(text_features.shape)




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda' 
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    main(device)
    