from Models.DeepVGAE import DeepVGAE,GCNEncoder
from Trainers import MMGNNTrainer

PROJECTION_DIM = 256

class VGAETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        gcn_encoder = GCNEncoder(PROJECTION_DIM,64,16)
        self.models['graph'] = DeepVGAE(gcn_encoder).to(self.device)

        loss = model.recon_loss(z, train_data.pos_edge_label_index) + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()


    def train_epoch():
        pass