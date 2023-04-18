
import argparse

from Trainers import MMGNNTrainer,MMGATTrainer,MMSAGETrainer,VGAETrainer,ClassifierTrainer,GMAETrainer,SuperGATTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', help='resume from checkpoint')
    parser.add_argument('--cpu', '-c', action='store_true',help='Use CPU only')
    parser.add_argument('--workers', '-w',default=2, type=int,help='no of workers')    
    parser.add_argument('--epochs', '-e',default=2, type=int,help='Epochs')
    parser.add_argument('--optim', '-o',default='SGD', type=str,help='optimizer type')
    parser.add_argument('--batchsize', '-bs',default=8, type=int,help='Batch Size')
    parser.add_argument('--model', '-m',default='MMGCN', type=str)
    # parser.add_argument('--task', '-t',default='train_e_e', type=str,help='')
    parser.add_argument('--data_path', '-d',default='./data', type=str)
    parser.add_argument('--dataset', '-ds',default='HatefulMeme', type=str)
    
    args = parser.parse_args()
    
    if args.model=='MMGCN':
        net = MMGNNTrainer(args)
    elif args.model=='MMGAT':
        net = MMGATTrainer(args)
    elif args.model=='MMSAGE':
        net = MMSAGETrainer(args)
    elif args.model=="VGAE_UNSUPERVISED":
        net = VGAETrainer(args)
    elif args.model=="MLP_EMBED_CLASSIFIER":
        net = ClassifierTrainer(args)
    elif args.model=="GMAE":
        net = GMAETrainer(args)
    elif args.model=="SUPERGAT":
        net = SuperGATTrainer(args)
    print(args.model,"============================================")
    net.train()

    
    print("Model Training Completed")
