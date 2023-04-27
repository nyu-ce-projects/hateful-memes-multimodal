import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear


class GAT(torch.nn.Module):
    def __init__(self,num_features,num_classes=1,training=True):
        super(GAT, self).__init__()
        self.training = training
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, 16, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.classifier = Linear(16, num_classes)
    
    def forward(self,x, edge_index, batch):    
        # Dropout before the GAT layer helps avoid overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=True)
        out = self.classifier(x)
        out = out.view(-1)
        return out,x    #,F.log_softmax(x, dim=1)