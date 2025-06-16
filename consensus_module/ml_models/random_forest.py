from sklearn.ensemble import RandomForestClassifier

class BuildRandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=0)

    def train_model(self, data_x, data_y):
        self.model.fit(data_x, data_y)

    def get_trained_model(self):
        return self.model