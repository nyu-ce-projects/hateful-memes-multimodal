import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
from Trainers.BaseTrainer import BaseTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GCN import GCN,GCNClassifier
from transformers import AutoTokenizer
from Dataset.HatefulMemeDataset import HatefulMemeDataset,collate_fn

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GDataLoader

PROJECTION_DIM = 256

class CLIPGNNTrainer(BaseTrainer):
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
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)

        dev_dataset = HatefulMemeDataset('./data','dev',image_transform,tokenizer)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)

        test_dataset = HatefulMemeDataset('./data','test',image_transform,tokenizer)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)

    def build_model(self):
        # Model
        print('==> Building model..')
        self.models['image_encoder'] = ImageEncoder().to(self.device)
        self.models['text_encoder'] = TextEncoder().to(self.device)
        self.models['image_projection'] = ProjectionHead(2048,PROJECTION_DIM).to(self.device)
        self.models['text_projection'] = ProjectionHead(768,PROJECTION_DIM).to(self.device)
        self.models['graph'] = GCNClassifier(PROJECTION_DIM,2).to(self.device)
        self.imgfeatureModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        if self.device in ['cuda','mps']:
            for key, model in self.models.items():
                self.models[key] = torch.nn.DataParallel(model)
                cudnn.benchmark = True

    def train_epoch(self,epoch):
        self.setTrain()
        train_loss = 0
        total = 0
        preds = []
        proba = []
        out_label_ids = []
        for images, tokenized_text, attention_masks, labels in self.train_loader:
            images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_embeddings = self.get_image_feature_embeddings(images)
            g_data = self.generate_subgraph(images,image_embeddings,text_embeddings)

            outputs = self.models['graph'](g_data.x,g_data.edge_index)

            loss = self.criterion(outputs[0], labels) #using nll_loss
            loss.backward()
            

            self.optimizer.step()
            
            # Metrics Calculation
            preds = np.append(preds, torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5, axis=0)
            proba = np.append(proba, torch.sigmoid(outputs[0]).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

            train_loss += loss.item()
            total += labels.size(0)
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
        preds = []
        proba = []
        out_label_ids = []
        with torch.no_grad():
            for images, tokenized_text, attention_masks, labels in self.dev_loader:
                images, tokenized_text, attention_masks, labels = images.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))

                g_data = self.generate_subgraph(image_embeddings,text_embeddings)

                outputs = self.models['graph'](g_data.x,g_data.edge_index)

                loss = self.criterion(outputs[0], labels)

                # Metrics Calculation
                test_loss += loss.item()
                total += labels.size(0)
                preds = np.append(preds, torch.sigmoid(outputs[0]).detach().cpu().numpy() > 0.5, axis=0)
                proba = np.append(proba, torch.sigmoid(outputs[0]).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

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
            # masks.append(output['masks']*imgTensors[i])
            embd = self.models['image_projection'](self.models['image_encoder'](output['masks']*imgTensors[i]))
            embeddings.append(embd)
        
        return embeddings


    def generate_subgraph(self,images,image_embeddings,text_embeddings):
        data_list = []
        for i in range(len(image_embeddings)):
            data = HeteroData().to(self.device)

            data['image_node'].node_id = [0]
            data['text_embeddings'].x = text_embeddings[i].unsqueeze(0)
            data['image_feature_embeddings'].x = image_embeddings[i]
            n_img_features = len(image_embeddings[i])
            data['image_node','has','image_feature_embeddings'].edge_index = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)
            data['text_embeddings','associated','image_feature_embeddings'].edge_index = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)

            data.validate(raise_on_error=True)
            data_list.append(data)
        loader = GDataLoader(data_list, batch_size=self.batch_size)
        return loader