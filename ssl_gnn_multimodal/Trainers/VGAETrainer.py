import os
from Models.DeepVGAE import DeepVGAE,GCNVGAEEncoder,GATVGAEEncoder
from Trainers import MMGNNTrainer
from Models.GraphClassifier import GraphClassifier

import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import MLP, MLPAggregation,SetTransformerAggregation,DeepSetsAggregation,GRUAggregation
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from config import PROJECTION_DIM


class VGAETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_model(self):
        super().build_model()
        self.trainable_models = ['image_encoder','text_encoder','image_projection','text_projection','graph','gnn_encoder']
        self.models['gnn_encoder'] = GATVGAEEncoder(PROJECTION_DIM,512,1024,4,0.3)
        self.models['graph'] = DeepVGAE(self.models['gnn_encoder']).to(self.device)
        if self.pretrain is not True:
            max_num_nodes_in_graph = 12
            self.models['readout_aggregation'] = MLPAggregation(1024,1024,max_num_nodes_in_graph,num_layers=1)
            self.models['classifier'] = GraphClassifier(1024,1, 2,self.models['readout_aggregation'], True,0.5).to(self.device)
            self.trainable_models = ['gnn_encoder','graph','readout_aggregation','classifier']

    

    def train_epoch(self,epoch):
        self.setTrain(self.trainable_models)
        train_loss = 0
        total = 0
        for images, tokenized_text, attention_masks, labels in self.train_loader:
            images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_embeddings = self.get_image_feature_embeddings(images)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
            
            g_data = next(iter(g_data_loader))
            g_data = g_data.to(self.device)
            
            z = self.models['graph'].encode(g_data.x, g_data.edge_index)
            
            if self.pretrain is True:
            # Pretraining  
                loss = self.models['graph'].loss(z, g_data)
            else:
            # hateful classification
                output = self.models['classifier'](z,g_data)
                loss = self.criterion(output, g_data.y)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)

        
        print("Training --- Epoch : {} | Loss : {}".format(epoch,train_loss/total))    
        return epoch,train_loss/total
    
    def evaluate(self, epoch, data_type, data_loader):
        self.setEval()
        test_loss = 0
        total = 0
        roc_auc_scores = []
        ap_scores = []
        micro_f1s = []
        accuracies = []
        with torch.no_grad():
            for images, tokenized_text, attention_masks, labels in data_loader:
                images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_embeddings = self.get_image_feature_embeddings(images)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
                

                g_data = next(iter(g_data_loader))
                g_data = g_data.to(self.device)

                z = self.models['graph'].encode(g_data.x, g_data.edge_index)
                
                if self.pretrain is True:
                # Pretraining  
                    loss = self.models['graph'].loss(z, g_data)
                    roc_auc,ap_score = self.models['graph'].metrics(z, g_data)
                    
                else:
                # hateful classification
                    output = self.models['classifier'](z,g_data)
                    loss = self.criterion(output, g_data.y)
                    # Metrics Calculation
                    roc_auc,ap_score,micro_f1,accuracy = self.models['classifier'].metrics(output,labels)
                    micro_f1s.append(micro_f1)
                    accuracies.append(accuracy)

                test_loss += loss.item()
                total += labels.size(0)

                roc_auc_scores.append(roc_auc)
                ap_scores.append(ap_score)

        metrics = {
            "loss": test_loss/total,
            "auc": round(np.mean(roc_auc_scores),3),
            "avg_precision": round(np.mean(ap_scores),3)
        }
        if len(accuracies)>0:
            metrics['accuracy'] = round(np.mean(accuracies),3)
        
        if len(micro_f1s)>0:
            metrics['micro_f1'] = round(np.mean(micro_f1s),3)

        print("{} --- Epoch : {} | roc_auc_score : {} | average_precision_score : {}".format(data_type,epoch,metrics['auc'],metrics['avg_precision']))    
        return metrics                       

    def save_checkpoint(self,epoch, metrics):
        training_type = "classifier"
        if self.pretrain:
            training_type = "pretrain"
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}_{}".format(training_type,metrics['auc'],metrics['avg_precision']))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                
                    print('Saving..')
                    for name, model in self.models.items():
                        savePath = os.path.join(outpath, "{}.pth".format(name))
                        toSave = model.state_dict()
                        torch.save(toSave, savePath)
                    savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                    torch.save(self.optimizer.state_dict(), savePath)
                    # self.best_acc = metrics['accuracy']
                    self.best_auc = metrics['auc']
                    print("best auc:", metrics['auc'],"avg_precision:",metrics['avg_precision'])
        except Exception as e:
            print("Error:",e)
