import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, num_node_features]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,0.1)
        x = self.conv2(x, edge_index)
        return x,F.log_softmax(x, dim=1)

class GCNClassifier(torch.nn.Module):
    def __init__(self,in_channels=256,num_classes=2):
        super().__init__()
        # torch.manual_seed(1234)
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 16)
        self.classifier = Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # 2. Readout layer
        h = global_mean_pool(h, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        h = F.dropout(h, p=0.5, training=True)
        out = self.classifier(h)
        out = out.view(-1)
        return out, h,