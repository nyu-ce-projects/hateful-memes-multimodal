import os
import torch
# import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
from Trainers.BaseTrainer import BaseTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GCN import GCN,GCNClassifier
from transformers import AutoTokenizer,DistilBertTokenizer
from Dataset.HatefulMemeDataset import HatefulMemeDataset

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
from torch_geometric.data import HeteroData,Data as GraphData,Batch
from torch_geometric.loader import DataLoader as GDataLoader,DataListLoader
from torch_geometric.nn import to_hetero,DataParallel
import torch_geometric.transforms as T

import numpy as np

PROJECTION_DIM = 256

class MMGNNTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.load_dataset()
        self.build_model()
        
        self.getTrainableParams()
        self.setup_optimizer_losses()
        if args.resume:
            self.load_checkpoint()

    def build_model(self):
        # Model
        print('==> Building model..')
        self.models = {
            'image_encoder': ImageEncoder().to(self.device),
            'text_encoder': TextEncoder().to(self.device),
            'image_projection': ProjectionHead(2048,PROJECTION_DIM).to(self.device),
            'text_projection': ProjectionHead(768,PROJECTION_DIM).to(self.device),
            'graph': GCNClassifier(PROJECTION_DIM,1).to(self.device)
        }
        self.imgfeatureModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.enable_multi_gpu()

    def train_epoch(self,epoch):
        self.setTrain()
        train_loss = 0
        total = 0
        preds = None
        proba = None
        out_label_ids = None
        for images, tokenized_text, attention_masks, labels in self.train_loader:
            images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            # images, image_features, text_features,labels
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_embeddings = self.get_image_feature_embeddings(images)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
            
            # Hetero Data : TODO
            # self.models['graph'] = to_hetero(self.models['graph'],g_data.metadata(),aggr='sum')
            # outputs = self.models['graph'](g_data.x_dict,g_data.edge_index_dict)
        
            g_data = next(iter(g_data_loader))
            # for g_data in g_data_loader:
            g_data = g_data.to(self.device)
            outputs = self.models['graph'](g_data.x,g_data.edge_index,g_data.batch)

            loss = self.criterion(outputs[0], g_data.y)
            loss.backward()

            self.optimizer.step()
            
            # Metrics Calculation
            if preds is None:
                preds = torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5
            else:
                preds = np.append(preds, torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5, axis=0)
            if proba is None:
                proba = torch.sigmoid(outputs[0]).detach().cpu().numpy()
            else:
                proba = np.append(proba, torch.sigmoid(outputs[0]).detach().cpu().numpy(), axis=0)
            
            if out_label_ids is None:
                out_label_ids = labels.detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

            train_loss += loss.item()
            total += labels.size(0)

        metrics =  {
            "loss": train_loss/total,
            "accuracy": round(accuracy_score(out_label_ids, preds),3),
            "auc": round(roc_auc_score(out_label_ids, proba),3),
            "micro_f1": round(f1_score(out_label_ids, preds, average="micro"),3)
        }
        print("Training --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {}".format(epoch,metrics['accuracy'],metrics['loss'],metrics['auc']))    
        return metrics

    def evaluate(self, epoch, data_type, data_loader):
        self.setEval()
        test_loss = 0
        total = 0
        preds = None
        proba = None
        out_label_ids = None
        with torch.no_grad():
            for images, tokenized_text, attention_masks, labels in data_loader:
                images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_embeddings = self.get_image_feature_embeddings(images)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
                

                g_data = next(iter(g_data_loader))
                # for g_data in g_data_loader:
                g_data = g_data.to(self.device)
                outputs = self.models['graph'](g_data.x,g_data.edge_index,g_data.batch)
                loss = self.criterion(outputs[0], g_data.y)
                test_loss += loss.item()

                # Metrics Calculation
                if preds is None:
                    preds = torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5
                else:
                    preds = np.append(preds, torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5, axis=0)
                if proba is None:
                    proba = torch.sigmoid(outputs[0]).detach().cpu().numpy()
                else:
                    proba = np.append(proba, torch.sigmoid(outputs[0]).detach().cpu().numpy(), axis=0)
                
                if out_label_ids is None:
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

                total += labels.size(0)

        result =  {
            "loss": test_loss/total,
            "accuracy": round(accuracy_score(out_label_ids, preds),3),
            "auc": round(roc_auc_score(out_label_ids, proba),3),
            "micro_f1": round(f1_score(out_label_ids, preds, average="micro"),3),
            "prediction": preds
        }
        print("{} --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {}".format(data_type, epoch,result['accuracy'],result['loss'],result['auc']))    
        return result

    def get_image_feature_embeddings(self,imgTensors):
        embeddings = []
        outputs = self.imgfeatureModel(imgTensors)
        for i,output in enumerate(outputs):
            indices = torch.argsort(output['scores'])[-10:] #get top 10 features
            masks = output['masks'][indices]
            embd = self.models['image_projection'](self.models['image_encoder'](masks*imgTensors[i]))
            embeddings.append(embd)
        
        return embeddings

    def generate_hetero_subgraph(self,images,image_embeddings,text_embeddings):
        data_list = []
        for i in range(len(image_embeddings)):
            data = HeteroData().to(self.device)
            # [41,1,256,256] => [[41, 3,256,256]]
            # data['image_node'].node_id = [0]
            data['text_embeddings'].x = text_embeddings[i].unsqueeze(0)
            data['image_feature_embeddings'].x = image_embeddings[i]
            n_img_features = len(image_embeddings[i])
            # data['image_node','has','image_feature_embeddings'].edge_index = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)
            data['text_embeddings','associated','image_feature_embeddings'].edge_index = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)

            # data.validate(raise_on_error=True)
            # data_list.append(data)
        # loader = GDataLoader(data_list, batch_size=self.batch_size)
        return data

    def generate_subgraph(self,image_embeddings,image_feat_embeddings,text_embeddings,labels):
        data_list = []
        for i in range(len(image_embeddings)):
            n_img_features = len(image_feat_embeddings[i])
            data = GraphData().to(self.device)
            data.x = torch.cat([image_embeddings[i].unsqueeze(0),text_embeddings[i].unsqueeze(0),image_feat_embeddings[i]])
            

            imgEdges = torch.tensor([[0]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)
            textEdges = torch.tensor([[1]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)

            data.edge_index = torch.cat([imgEdges,textEdges],dim=1)
            data.y = labels[i]
            data = T.ToUndirected()(data)
            data = T.NormalizeFeatures()(data)
            data_list.append(data)
        
        loader = GDataLoader(data_list, batch_size=self.batch_size*self.n_gpus)
        # return Batch().from_data_list(data_list)
        return loader

    def load_checkpoint(self):
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint_dir = self.args.resume
            print(checkpoint_dir)
            assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'

            model_keys = ['image_encoder','text_encoder','image_projection','text_projection','graph']
            for key in model_keys:
                model_path = os.path.join(checkpoint_dir,"{}.pth".format(key))
                checkpoint = torch.load(model_path,map_location=self.device)
                remove_prefix = 'module.'
                state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint.items()}
                self.models[key].load_state_dict(state_dict)
            checkpoint = torch.load(os.path.join(checkpoint_dir,"{}.pth".format(self.optim.lower())))
            self.optimizer.load_state_dict(checkpoint)
        except Exception as e:
            print(e)