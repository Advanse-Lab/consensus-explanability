import pandas as pd
from IPython.display import display

from folktables import ACSIncome
from sklearn.model_selection import train_test_split

from consensus_module.ml_models.random_forest import BuildRandomForest
from consensus_module.ml_models.logistic_regression import BuildLogisticRegression

class MlModel:
    def __init__(self, dataset_name, id_column, target_column, exception = None):
        # importing training dataset
        self.data = pd.read_csv(dataset_name)

        if id_column:
            self.data = self.data.set_index(id_column)

        if target_column:
            self.data = self.data.dropna(subset=[target_column])
            # splitting target column
            self.data_x = self.data.drop([target_column], axis=1)
            self.data_y = self.data[target_column]

        if exception:
            features, label, _ = ACSIncome.df_to_pandas(self.data)

            self.data_x, self.test_x, self.data_y, self.test_y = train_test_split(features,label,train_size=0.7)
    
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
    
    def getXTest(self):
        return self.test_x
    
    def show_train_dataset(self):
        # showing train dataframe
        display(self.data)