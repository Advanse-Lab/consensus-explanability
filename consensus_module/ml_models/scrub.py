from skrub import tabular_learner

class BuildScrub:
    def __init__(self):
        self.model=tabular_learner('classifier')

    def train_model(self, data_x, data_y):
        self.model.fit(data_x, data_y)

    def get_trained_model(self):
        return self.model