import os
import torch
from Trainers.MMGNNTrainer import MMGNNTrainer
from Models.DeepVGAE import DeepVGAE,GCNEncoder
from Models.GraphClassifier import GraphClassifier,AdaptiveReadoutMLPClassifier
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

PROJECTION_DIM = 256

class ClassifierTrainer(MMGNNTrainer):
    def __init__(self,args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        if args.resume:
            self.load_checkpoint()

    
    def build_model(self):
        super().build_model()
        gcn_encoder = GCNEncoder(PROJECTION_DIM,64,16)
        self.models['graph'] = DeepVGAE(gcn_encoder).to(self.device)
        # self.models['classifier'] = GraphClassifier(16,1, 2, True,0.5).to(self.device)
        self.models['classifier'] = AdaptiveReadoutMLPClassifier(16,8,1,1,num_layers=2).to(self.device)
    
    def train_epoch(self, epoch):
        self.setTrain()
        train_loss = 0
        total = 0
        preds = None
        proba = None
        out_label_ids = None
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
            
            output = self.models['classifier'](z,g_data)
            print(output.size())
            loss = self.criterion(output, g_data.y)
            loss.backward()

            self.optimizer.step()

            # Metrics Calculation
            if preds is None:
                preds = torch.sigmoid(output).detach().cpu().numpy() > 0.5
            else:
                preds = np.append(preds, torch.sigmoid(output).detach().cpu().numpy() > 0.5, axis=0)
            if proba is None:
                proba = torch.sigmoid(output).detach().cpu().numpy()
            else:
                proba = np.append(proba, torch.sigmoid(output).detach().cpu().numpy(), axis=0)
            
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
                g_data = g_data.to(self.device)
                z = self.models['graph'].encode(g_data.x, g_data.edge_index)
                output = self.models['classifier'](z,g_data)
                loss = self.criterion(output, g_data.y)
                test_loss += loss.item()

                # Metrics Calculation
                if preds is None:
                    preds = torch.sigmoid(output).detach().cpu().numpy() > 0.5
                else:
                    preds = np.append(preds, torch.sigmoid(output).detach().cpu().numpy() > 0.5, axis=0)
                if proba is None:
                    proba = torch.sigmoid(output).detach().cpu().numpy()
                else:
                    proba = np.append(proba, torch.sigmoid(output).detach().cpu().numpy(), axis=0)
                
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
            "prediction": preds,
            "labels": out_label_ids,
            "proba": proba
        }
        print("{} --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {}".format(data_type, epoch,result['accuracy'],result['loss'],result['auc']))    
        return result
    
