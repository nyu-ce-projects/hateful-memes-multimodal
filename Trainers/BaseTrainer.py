
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
        self.criterion = nn.CrossEntropyLoss()
        if self.optim=='SGD':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        elif self.optim=='SGDN':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4,nesterov=True)
        else:
            self.optimizer = eval("optim."+self.optim)(self.trainableParameters, lr=self.lr, weight_decay=0)
        print(self.optimizer) 
        # num_warmup_steps = self.warmup_epochs * len(self.poison_train_loader)
        # num_training_steps = (self.warmup_epochs+self.epochs) * len(self.poison_train_loader)
        # self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)  # TODO: try get_polynomial_decay_schedule_with_warmup,cosine
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
                self.evaluate(epoch)
                self.scheduler.step()
                # self.saveModel(model_version_name,epoch)
                print('*' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
             
    def train_epoch(self,epoch):
        raise NotImplementedError

    def evaluate(self,epoch):
        raise NotImplementedError
        # self.net.eval()
        # test_loss = 0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for padded_text, attention_masks, labels in dataloader:
        #         padded_text, attention_masks, labels = padded_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
        #         outputs = self.net(padded_text,attention_masks)
        #         loss = self.criterion(outputs, labels)

        #         test_loss += loss.item()
        #         _, predicted = outputs.max(1)
        #         total += labels.size(0)
        #         correct += predicted.eq(labels).sum().item()

        # acc = 100.*correct/total
            
        # return acc,test_loss/total
    
    # def saveCheckpoint(self,epoch,acc):
    #     # Save checkpoint.
    #     if acc > self.best_acc:
    #         print('Saving..')
    #         state = {
    #             'net': self.net.state_dict(),
    #             'acc': acc,
    #             'epoch': epoch,
    #         }
    #         if not os.path.isdir('checkpoint'):
    #             os.mkdir('checkpoint')
    #         torch.save(state, './checkpoint/ckpt.pth')
    #         self.best_acc = acc

    # def saveModel(self):
    #     outpath = os.path.join(self.config['model_path'],self.config['model_name'], "weights_{}".format(self.epoch))
    #     if not os.path.exists(outpath):
    #         os.makedirs(outpath)
    #     for name, model in self.models.items():
    #         savePath = os.path.join(outpath, "{}.pth".format(name))
    #         toSave = model.state_dict()
    #         if name == "encoder":
    #             toSave["height"] = self.height
    #             toSave["width"] = self.width
    #         torch.save(toSave, savePath)
    #     savePath = os.path.join(outpath, "adam.pth")
    #     torch.save(self.optimizer.state_dict(), savePath)