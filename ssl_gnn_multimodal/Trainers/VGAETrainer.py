import os
import math
from Models.DeepVGAE import DeepVGAE,GCNEncoder,GATEncoder
from Trainers import MMGNNTrainer
import torch
import torchvision
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from transformers import AutoTokenizer,DistilBertTokenizer
from Dataset.ConceptualCaptionDataset import ConceptualCaptionDataset

PROJECTION_DIM = 256

class VGAETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    # def load_dataset(self):
    #     # Data
    #     print('==> Preparing data..')
    #     image_transform = torchvision.transforms.Compose(
    #         [
    #             torchvision.transforms.Resize(size=(224, 224)),
    #             torchvision.transforms.ToTensor()
    #         ]
    #     )
    #     model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    #     dataset = ConceptualCaptionDataset(self.data_path,image_transform,tokenizer)
    #     n_dataset = len(dataset)
    #     n_dev = math.ceil(len(dataset)*0.05)
    #     n_test = math.ceil(len(dataset)*0.05)
    #     n_train = n_dataset - n_test - n_dev
    #     train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_dev, n_test))
        
    #     self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=dataset.collate_fn)

    #     self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=dataset.collate_fn)

    #     self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=dataset.collate_fn)


    def build_model(self):
        super().build_model()
        self.models['gnn_encoder'] = GATEncoder(PROJECTION_DIM,64,16,8,0.3)
        self.models['graph'] = DeepVGAE(self.models['gnn_encoder']).to(self.device)

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
            
            
            z = self.models['graph'].encode(g_data.x, g_data.edge_index)
            loss = self.models['graph'].recon_loss(z, g_data.edge_index) + (1 / g_data.num_nodes) * self.models['graph'].kl_loss()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)

        
        print("Training --- Epoch : {} | Loss : {}".format(epoch,train_loss/total))    
        return epoch,train_loss/total
    
    def evaluate(self, epoch, data_type, data_loader):
        self.setEval()
        roc_auc_scores = []
        ap_scores = []
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

                neg_edge_index = negative_sampling(g_data.edge_index, z.size(0))
                # metrics = self.models['graph'].test(z, g_data.edge_index, neg_edge_index)
                pos_y = z.new_ones(g_data.edge_index.size(1))
                neg_y = z.new_zeros(neg_edge_index.size(1))
                y = torch.cat([pos_y, neg_y], dim=0)

                pos_pred = self.models['graph'].decoder(z, g_data.edge_index, sigmoid=True)
                neg_pred = self.models['graph'].decoder(z, neg_edge_index, sigmoid=True)
                pred = torch.cat([pos_pred, neg_pred], dim=0)

                y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
                
                roc_auc,ap_score =  roc_auc_score(y, pred), average_precision_score(y, pred)
                roc_auc_scores.append(roc_auc)
                ap_scores.append(ap_score)


        print("{} --- Epoch : {} | roc_auc_score : {} | average_precision_score : {}".format(data_type,epoch,np.mean(roc_auc_scores),np.mean(ap_scores)))    
        return {
            "auc":np.mean(roc_auc_scores),
            "avg_precision":np.mean(ap_scores)
        }

    def save_checkpoint(self,epoch, metrics):
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}".format(metrics['auc'],metrics['avg_precision']))
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
