
import os
import torch
import torchvision
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

from Trainers import MMGNNTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GMAE import GMAE
from Models.GAT import GAT

PROJECTION_DIM = 256
class GMAETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_model(self):
        # Model
        print('==> Building model..')
        self.models = {
            'image_encoder': ImageEncoder().to(self.device),
            'text_encoder': TextEncoder().to(self.device),
            'image_projection': ProjectionHead(2048,PROJECTION_DIM).to(self.device),
            'text_projection': ProjectionHead(768,PROJECTION_DIM).to(self.device),
            'graph_encoder': GAT(in_channels=PROJECTION_DIM,hidden_channels=512,out_channels=1024,num_layers=3,in_heads=4,out_heads=1).to(self.device),
            'graph_decoder':GAT(in_channels=1024,hidden_channels=512,out_channels=PROJECTION_DIM,num_layers=1,in_heads=1,out_heads=1).to(self.device)
        }
        self.models['graph'] = GMAE(self.models['graph_encoder'],self.models['graph_decoder']).to(self.device)
        self.imgfeatureModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.enable_multi_gpu()

    def train_epoch(self,epoch):
        self.setTrain()
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
            
            loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
            
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
        acc_scores = []
        micro_f1s = []
        with torch.no_grad():
            for images, tokenized_text, attention_masks, labels in data_loader:
                images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_embeddings = self.get_image_feature_embeddings(images)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
                
                g_data = next(iter(g_data_loader))
                g_data = g_data.to(self.device)
                
                loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
                
                z = self.models['graph'].encode(g_data.x, g_data.edge_index)
                neg_edge_index = negative_sampling(g_data.edge_index, z.size(0))
                pos_y = z.new_ones(g_data.edge_index.size(1))
                neg_y = z.new_zeros(neg_edge_index.size(1))
                y = torch.cat([pos_y, neg_y], dim=0)

                pos_pred = self.models['graph'].decoder(z, g_data.edge_index, sigmoid=True)
                neg_pred = self.models['graph'].decoder(z, neg_edge_index, sigmoid=True)
                pred = torch.cat([pos_pred, neg_pred], dim=0)

                y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
                
                roc_auc,ap_score,acc_score,micro_f1 =  roc_auc_score(y, pred), average_precision_score(y, pred),accuracy_score(y,pred),f1_score(y, pred, average="micro")

                roc_auc_scores.append(roc_auc)
                ap_scores.append(ap_score)
                acc_scores.append(acc_score)
                micro_f1s.append(micro_f1)
                test_loss += loss.item()
                total += labels.size(0)

        metrics =  {
            "loss": test_loss/total,
            "accuracy": round(np.mean(acc_scores),3),
            "auc": round(np.mean(roc_auc_scores),3),
            "micro_f1": round(np.mean(micro_f1s),3),
            "avg_precision": round(np.mean(ap_scores),3),
        }
        print("{} --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {} | Micro-F1 : {}".format(data_type, epoch,metrics['accuracy'],metrics['loss'],metrics['auc'],metrics['micro_f1']))    
        return metrics

    def save_checkpoint(self,epoch, metrics):
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}".format(metrics['auc'],metrics['micro_f1']))
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