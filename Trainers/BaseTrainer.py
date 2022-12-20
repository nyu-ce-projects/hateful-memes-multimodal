
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os

import time



class BaseTrainer():
    def __init__(self,args) -> None:
        print(args)
        self.args = args
        self.model_name = 'mmgcn'
        self.lr = self.args.lr
        self.optim = self.args.optim
        self.num_workers = args.workers
        self.epochs = self.args.epochs
        self.batch_size = self.args.batchsize
        self.n_gpus = 1
        self.best_acc = 0
        self.set_device()
        
    def getTrainableParams(self):
        self.totalTrainableParams = 0
        self.trainableParameters = []
        for key in self.models:
            self.trainableParameters += list(self.models[key].parameters())
            self.totalTrainableParams += sum(p.numel() for p in self.models[key].parameters() if p.requires_grad)    

    def set_device(self):
        if self.args.cpu is not False:
            self.device = 'cpu'
        else:
            if torch.cuda.is_available():
                self.device = 'cuda' 
                self.n_gpus = torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            
        print(self.device)

    def load_dataset(self):
        raise NotImplementedError
        
    def setup_optimizer_losses(self):
        self.criterion = nn.BCEWithLogitsLoss()  #BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        if self.optim=='SGD':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        elif self.optim=='SGDN':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4,nesterov=True)
        else:
            self.optimizer = eval("optim."+self.optim)(self.trainableParameters, lr=self.lr, weight_decay=5e-4)
        print("Optimizer:",self.optimizer) 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def setTrain(self):
        for model in self.models.values():
            model.train()

    def setEval(self):
        for model in self.models.values():
            model.eval()

    def build_model(self):
        raise NotImplementedError

    def train(self):
        try:
            print("Total Trainable Parameters : {}".format(self.totalTrainableParams))
            model_version_name = int(time.time())
            for epoch in range(self.epochs):
                self.train_epoch(epoch)
                metrics = self.evaluate(epoch,self.dev_loader)
                self.scheduler.step()
                self.save_checkpoint(model_version_name,metrics['AUC'])
                print('*' * 89)
            unseen_metrics = self.evaluate(epoch, self.test_loader)
            print(unseen_metrics)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
             
    def train_epoch(self,epoch):
        raise NotImplementedError

    def evaluate(self,epoch):
        raise NotImplementedError

    def save_checkpoint(self,model_version_name,acc):
        try:
            outpath = os.path.join('./checkpoints',self.model_name, str(model_version_name))
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if acc > self.best_acc:
                print('Saving..')
                for name, model in self.models.items():
                    savePath = os.path.join(outpath, "{}.pth".format(name))
                    toSave = model.state_dict()
                    torch.save(toSave, savePath)
                savePath = os.path.join(outpath, "optimizer.pth")
                torch.save(self.optimizer.state_dict(), savePath)
                self.best_acc = acc
                print("best accuracy:", acc)
        except Exception as e:
            print("Error:",e)

    # def load_checkpoint(self):
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     checkpoint_dir = os.path.join('./checkpoints',self.model_name, str(1671488899))
    #     print(checkpoint_dir)
    #     # assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/{}/ckpt.pth'.format(self.device_name))
    #     self.net.load_state_dict(checkpoint['net'],strict=False)
    #     self.optimizer.load_state_dict(checkpoint['optmizer'],strict=False)
    #     self.best_acc = checkpoint['acc']
    #     self.epoch = checkpoint['epoch']