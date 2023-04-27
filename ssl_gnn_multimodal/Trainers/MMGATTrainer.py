
from Models.GAT import GATClassifier
from Trainers import MMGNNTrainer
from config import PROJECTION_DIM

class MMGATTrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        self.models['graph'] = GATClassifier(PROJECTION_DIM,num_classes=1).to(self.device)