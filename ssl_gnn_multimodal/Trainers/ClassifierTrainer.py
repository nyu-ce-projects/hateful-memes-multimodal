
    def run_downstream_classification(self):
        self.load_checkpoint()
        classifier_model = MLP(PROJECTION_DIM,1, 3, True,0.5)
