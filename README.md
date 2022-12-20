# Exploring Emerging Properties in Multimodal Neural Network 


### Dependencies
- Pytorch
- Torchvision
- torch-scatter
- torch-sparse
- pytorch geometric
- transformers
- timm
- tqdm


### Traing the Unimodal and Multimodal Networks
Notebook has been provided for these networks in the notebook directory.

### Training the Multimodal Graph Network
```
python main.py --batchsize 16 --epochs 1 --optim Adam --model MMSAGE --workers 4 --lr 0.01
```

Graph Models Available:
- GAT
- GCN
- GraphSage


|Modality| Model| AUROC| Accuracy|
|:----|:----|:----|:----|
|Unimodal| ResNet-50 (CASS)| 0.5405±0.045| 0.5445±0.015|
|Unimodal|Vit/Base-16 (CASS)| 0.53386±0.009| 0.59±0.017|
|Multimodal| ResNet-50 (CASS)| 0.501±0.078| 0.551±0.023|
|Multimodal| Vit/Base-16 (CASS)| 0.5±0.078| 0.625±0.032|
|Multimodal GNN| GCN| 0.514| 0.547|
|Multimodal GNN|GAT| 0.564| 0.499|
|Multimodal GNN|SAGE| 0.481| 0.62|



