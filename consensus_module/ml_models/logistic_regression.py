from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class BuildLogisticRegression:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(),LogisticRegression(solver="lbfgs",max_iter=10000))

    def train_model(self, data_x, data_y):
        self.model.fit(data_x, data_y)

    def get_trained_model(self):
        return self.model