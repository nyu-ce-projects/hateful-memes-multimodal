import torch
# import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
from Trainers.BaseTrainer import BaseTrainer
from Models import ImageEncoder,TextEncoder,ProjectionHead
from Models.GCN import GCN,GCNClassifier
from transformers import AutoTokenizer
from Dataset.HatefulMemeDataset import HatefulMemeDataset

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
from torch_geometric.data import HeteroData,Data as GraphData
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.nn import to_hetero

from tqdm import tqdm
import numpy as np

PROJECTION_DIM = 256

class MMGNNTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.load_dataset()
        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def load_dataset(self):
        # Data
        print('==> Preparing data..')
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()
            ]
        )
        model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        train_dataset = HatefulMemeDataset('./data','train',image_transform,tokenizer)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=train_dataset.collate_fn)

        dev_dataset = HatefulMemeDataset('./data','dev',image_transform,tokenizer)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=dev_dataset.collate_fn)

        test_dataset = HatefulMemeDataset('./data','test',image_transform,tokenizer)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=test_dataset.collate_fn)

    def build_model(self):
        # Model
        print('==> Building model..')
        self.models = {}
        self.models['image_encoder'] = ImageEncoder().to(self.device)
        self.models['text_encoder'] = TextEncoder().to(self.device)
        self.models['image_projection'] = ProjectionHead(2048,PROJECTION_DIM).to(self.device)
        self.models['text_projection'] = ProjectionHead(768,PROJECTION_DIM).to(self.device)
        self.models['graph'] = GCNClassifier(PROJECTION_DIM,1).to(self.device)
        self.imgfeatureModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        if self.device in ['cuda','mps']:
            for key, model in self.models.items():
                self.models[key] = torch.nn.DataParallel(model)
                # cudnn.benchmark = True

    def train_epoch(self,epoch):
        self.setTrain()
        train_loss = 0
        total = 0
        preds = None
        proba = None
        out_label_ids = None
        for images, tokenized_text, attention_masks, labels in self.train_loader:
            images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            print("Akash")
            self.optimizer.zero_grad()
            
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_embeddings = self.get_image_feature_embeddings(images)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
            
            # Hetero Data : TODO
            # self.models['graph'] = to_hetero(self.models['graph'],g_data.metadata(),aggr='sum')
            # outputs = self.models['graph'](g_data.x_dict,g_data.edge_index_dict)
        
            g_data = next(iter(g_data_loader))
            print(g_data)
            outputs = self.models['graph'](g_data.x,g_data.edge_index,g_data.batch)
            print(outputs[0],g_data.y)
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
            del g_data_loader
        result =  {
            "loss": train_loss/total,
            "accuracy": accuracy_score(out_label_ids, preds),
            "AUC": roc_auc_score(out_label_ids, proba),
            "micro_f1": f1_score(out_label_ids, preds, average="micro"),
            "prediction": preds,
            "labels": out_label_ids,
            "proba": proba
        }
        print("Training --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {}".format(epoch,result['accuracy'],result['loss'],result['AUC']))    
        return result

    def evaluate(self, epoch):
        self.setEval()
        test_loss = 0
        total = 0
        preds = None
        proba = None
        out_label_ids = None
        with torch.no_grad():
            for images, tokenized_text, attention_masks, labels in self.dev_loader:
                images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_embeddings = self.get_image_feature_embeddings(images)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph(image_embeddings,image_feat_embeddings,text_embeddings,labels)
                
                g_data = next(iter(g_data_loader))
                outputs = self.models['graph'](g_data.x,g_data.edge_index,g_data.batch)
                print(outputs[0],g_data.y)
                loss = self.criterion(outputs[0], g_data.y)

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

                test_loss += loss.item()
                total += labels.size(0)

        result =  {
            "loss": test_loss/total,
            "accuracy": accuracy_score(out_label_ids, preds),
            "AUC": roc_auc_score(out_label_ids, proba),
            "micro_f1": f1_score(out_label_ids, preds, average="micro"),
            "prediction": preds,
            "labels": out_label_ids,
            "proba": proba
        }
        print("Testing --- Epoch : {} | Accuracy : {} | Loss : {} | AUC : {}".format(epoch,result['accuracy'],result['loss'],result['AUC']))    
        return result

    def get_image_feature_embeddings(self,imgTensors):
        # masks = []
        embeddings = []
        outputs = self.imgfeatureModel(imgTensors)
        for i,output in enumerate(outputs):
            indices = torch.argsort(output['scores'])[-5:] #get top 10 features
            masks = output['masks'][indices]
            # masks.append(output['masks']*imgTensors[i])
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
            data_list.append(data)
        # print(len(data_list))
        loader = GDataLoader(data_list, batch_size=self.batch_size)
        return loader