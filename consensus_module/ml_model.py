import pandas as pd
from IPython.display import display

from consensus_module.ml_models.random_forest import BuildRandomForest
from consensus_module.ml_models.logistic_regression import BuildLogisticRegression

class MlModel:
    def __init__(self, dataset_name):
        # importing training dataset
        training_samples = pd.read_csv(dataset_name)
        self.data = training_samples.set_index("id_")

        # splitting target column
        self.data_x = self.data.drop(['y'], axis=1)
        self.data_y = self.data['y']
    
    def build_model(self, model_type="RF"):
        self.model_type = model_type
        # building and training random forest model
        if self.model_type == "RF":
            ml_model = BuildRandomForest()
            ml_model.train_model(self.data_x, self.data_y)
            
        elif self.model_type == "LR":
            ml_model = BuildLogisticRegression()
            ml_model.train_model(self.data_x, self.data_y)

        else:
            raise "No module build prepared with this name."

        self.ml_model = ml_model.get_trained_model()


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