import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv



class GraphSAGE(torch.nn.Module):
    ##GraphSAGE
    def __init__(self, dim_in, dim_h, dim_out,training=True):
        super().__init__()
        self.training = training
        self.sage1 = SAGEConv(dim_in, dim_h*2)
        self.sage2 = SAGEConv(dim_h*2, dim_h)
        self.sage3 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.sage3(h, edge_index)
        return h, F.log_softmax(h, dim=1)