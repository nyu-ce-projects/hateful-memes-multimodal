
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os

import time
from tqdm import tqdm
from Dataset import load_dataset
from torch.utils.data import DataLoader
from utils import get_device

class BaseTrainer():
    def __init__(self,args) -> None:
        print(args)
        self.args = args
        self.model_name = self.args.model
        self.lr = self.args.lr
        self.optim = self.args.optim
        self.num_workers = args.workers
        self.epochs = self.args.epochs
        self.batch_size = self.args.batchsize
        self.dataset_name = self.args.dataset
        self.data_path = self.args.data_path
        self.best_acc = 0
        self.best_auc = 0
        self.set_device()
        
    def getTrainableParams(self):
        self.totalTrainableParams = 0
        self.trainableParameters = []
        for key in self.models:
            self.trainableParameters += list(self.models[key].parameters())
            self.totalTrainableParams += sum(p.numel() for p in self.models[key].parameters() if p.requires_grad)    

    def set_device(self):
        self.n_gpus = 1
        if self.args.cpu is not False:
            self.device = 'cpu'
        else:
            self.device , self.n_gpus = get_device()
        print(self.device)

    def enable_multi_gpu(self):
        if self.device in ['cuda','mps'] and self.n_gpus>1:
            for key, model in self.models.items():
                if key!='graph':
                    self.models[key] = torch.nn.DataParallel(model)
                # else:
                #     self.models[key] = DataParallel(model)
                # cudnn.benchmark = True

    def load_dataset(self):
        train_dataset,dev_dataset,test_dataset,collate_fn = load_dataset(self.dataset_name,self.data_path) 
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)

        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)


    def setup_optimizer_losses(self):
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()  #BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        if self.optim=='SGD':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        elif self.optim=='SGDN':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4,nesterov=True)
        else:
            self.optimizer = eval("optim."+self.optim)(self.trainableParameters, lr=self.lr, weight_decay=5e-4)
        print("Optimizer:",self.optimizer) 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def setTrain(self,model_keys=[]):
        evalKeys = []
        if model_keys is not None and len(model_keys)>0:
            evalKeys = self.models.keys() - model_keys
        for model in self.models.values():
            model.train()
        
        for key in evalKeys:
            self.models[key].eval()

    def setEval(self):
        for model in self.models.values():
            model.eval()

    def build_model(self):
        raise NotImplementedError

    def train(self):
        try:
            print("Total Trainable Parameters : {}".format(self.totalTrainableParams))
            
            for epoch in tqdm(range(self.epochs)):
                self.train_epoch(epoch)
                metrics = self.evaluate(epoch,'Validation', self.dev_loader)
                self.scheduler.step()
                self.save_checkpoint(epoch,metrics)
                print('*' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        unseen_metrics = self.evaluate(epoch, 'Testing', self.test_loader)
        print(unseen_metrics)
             
    def train_epoch(self,epoch):
        raise NotImplementedError

    def evaluate(self,epoch):
        raise NotImplementedError

    def save_checkpoint(self,epoch, metrics):
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}".format(metrics['auc'],metrics['accuracy']))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                
                    print('Saving..')
                    print("Saved Model - Metrics",metrics)
                    for name, model in self.models.items():
                        savePath = os.path.join(outpath, "{}.pth".format(name))
                        toSave = model.state_dict()
                        torch.save(toSave, savePath)
                    savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                    torch.save(self.optimizer.state_dict(), savePath)
                    self.best_acc = metrics['accuracy']
                    self.best_auc = metrics['auc']
                    print("best auc:", metrics['auc'])
        except Exception as e:
            print("Error:",e)

    def load_checkpoint(self):
        raise NotImplementedError