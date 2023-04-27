from Models.SAGE import GraphSAGE
from Trainers import MMGNNTrainer

from config import PROJECTION_DIM

class MMSAGETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        self.models['graph'] = GraphSAGE(PROJECTION_DIM,64,1).to(self.device)