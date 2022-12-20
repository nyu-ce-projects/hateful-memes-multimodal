# Exploring Emerging Properties in Multimodal Neural Network 

There has been a sharp increase in the number of people using social media recently. Because of this, there has been an increase in the amount of content posted online. Manual moderation of content online is extremely difficult. To overcome this barrier, many automated moderation techniques have been designed. These techniques work with a singular modality (like text, audio, images, etc.) but suffer in a multi-modal environment where one modality complements one or more modalities implying subtle hate/aggressive speech. To address this problem, various unimodal, as well as multimodal approaches have been proposed. In general vision, only models perform worse than natural language models; with this study, we aim to address this by using novel, unimodal self-supervised techniques and further studying the scope of their expansion to make them fully multimodal. These multimodal self-supervised techniques are able to match the accuracy of fully supervised multimodally finetuned models.

[Paper](https://drive.google.com/drive/u/0/folders/1Kts1qrOYG_ZRTQncWdD95Q1QpAqvZqQS)

### Dependencies
- Pytorch
- Torchvision
- torch-scatter
- torch-sparse
- pytorch geometric
- transformers
- timm
- tqdm
- torchcontrib
- pytorch lightning


### Traing the Unimodal and Multimodal Networks
Notebook has been provided for these networks in the `notebook/` directory.



### Training the Multimodal Graph Network
```
python main.py --batchsize 16 --epochs 1 --optim Adam --model MMSAGE --workers 4 --lr 0.01
```

Graph Models Available:
- GAT
- GCN
- GraphSage

### Results

|Modality| Model| AUROC| Accuracy|
|:----|:----|:----|:----|
|Unimodal| ResNet-50 (CASS)| 0.5405±0.045| 0.5445±0.015|
|Unimodal|Vit/Base-16 (CASS)| 0.53386±0.009| 0.59±0.017|
|Multimodal| ResNet-50 (CASS)| 0.501±0.078| 0.551±0.023|
|Multimodal| Vit/Base-16 (CASS)| 0.5±0.078| 0.625±0.032|
|Multimodal GNN| GCN| 0.514| 0.547|
|Multimodal GNN|GAT| 0.564| 0.499|
|Multimodal GNN|SAGE| 0.481| 0.62|


### High Performance Optimization and Benchmark

Performance Benchmark details and script can be found in the `benchmarks/performance-hpc/`
