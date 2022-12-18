
import argparse

from Trainers import MMGNNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--cpu', '-c', action='store_true',help='Use CPU only')
    parser.add_argument('--workers', '-w',default=2, type=int,help='no of workers')    
    parser.add_argument('--epochs', '-e',default=2, type=int,help='Epochs')
    parser.add_argument('--optim', '-o',default='SGD', type=str,help='optimizer type')
    parser.add_argument('--batchsize', '-bs',default=32, type=int,help='Batch Size')
    args = parser.parse_args()
    
    net = MMGNNTrainer(args)
    net.train()
    
    print("Model Training Completed")
