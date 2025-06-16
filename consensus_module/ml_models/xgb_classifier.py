from xgboost import XGBClassifier

class BuildXGB:
    def __init__(self):
        self.model = XGBClassifier(max_depth=6, n_estimators = 200, random_state=8)

    def train_model(self, data_x, data_y):
        self.model.fit(data_x, data_y)

    def get_trained_model(self):
        return self.model