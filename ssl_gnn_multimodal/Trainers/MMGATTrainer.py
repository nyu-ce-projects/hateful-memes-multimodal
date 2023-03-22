
from Models.GAT import GAT
from Trainers import MMGNNTrainer

PROJECTION_DIM = 256

class MMGATTrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        self.models['graph'] = GAT(PROJECTION_DIM,num_classes=1).to(self.device)