import torch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from utils import get_normalization,get_activation



class GATClassifier(torch.nn.Module):
    def __init__(self,num_features,num_classes=1,training=True):
        super(GATClassifier, self).__init__()
        self.training = training
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATv2Conv(self.hid*self.in_head, 16, concat=False,
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
    

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers,in_heads,out_heads,norm_type="graph_norm",activation_type="prelu",dropout=0.3):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.in_head = in_heads
        self.out_heads = out_heads
        self.dropout = dropout

        normFn = get_normalization(norm_type)
        activation = get_activation(activation_type)
        self.gat_layers = []
        self.norms = []
        self.acts = []
        if num_layers == 1:
            self.gat_layers.append(GATv2Conv(in_channels, out_channels,out_heads,dropout=dropout))
            self.norms.append(torch.nn.Identity())
            self.acts.append(torch.nn.Identity())
        else:
            # input projection
            self.gat_layers.append(GATv2Conv(in_channels, hidden_channels,in_heads,dropout=dropout))
            self.norms.append(normFn(hidden_channels))
            self.acts.append(activation)
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATv2Conv(hidden_channels*in_heads, hidden_channels,in_heads,dropout=dropout))
                self.norms.append(normFn(hidden_channels))
                self.acts.append(activation)

            # output projection
            self.gat_layers.append(GATv2Conv(hidden_channels*in_heads, out_channels,out_heads,dropout=dropout))
            self.norms.append(torch.nn.Identity())
            self.acts.append(torch.nn.Identity())


        self.head = torch.nn.Identity()

    def forward(self, x, edge_index, return_hidden=False):
        h = x
        hidden_list = []
        for i, (gat_layer, norm, act) in enumerate(zip(self.gat_layers, self.norms,self.acts)):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = act(norm(gat_layer(h,edge_index)))
            hidden_list.append(h)

        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)