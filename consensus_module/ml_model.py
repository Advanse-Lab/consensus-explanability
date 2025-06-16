import pandas as pd
from IPython.display import display

from consensus_module.ml_models.random_forest import BuildRandomForest

class MlModel:
    def __init__(self, dataset_name):
        # importing training dataset
        training_samples = pd.read_csv(dataset_name)
        self.data = training_samples.set_index("id_")

        # splitting target column
        self.data_x = self.data.drop(['y'], axis=1)
        self.data_y = self.data['y']
    
    def build_model(self, type="RF"):
        # building and training random forest model
        if type == "RF":
            rf_model = BuildRandomForest()
            rf_model.train_model(self.data_x, self.data_y)
            self.ml_model = rf_model.get_trained_model()

    def get_feature_names(self):
        # getting list of feature names
        return list(self.data_x.columns)
    
    def getMlModel(self):
        try:
            return self.ml_model
        except:
            raise "No model built yet."
    
    def getXData(self):
        return self.data_x
    
    def getYData(self):
        return self.data_y
    
    def show_train_dataset(self):
        # showing train dataframe
        display(self.data)