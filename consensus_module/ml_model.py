import pandas as pd
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier

class MlModel:
    def __init__(self, dataset_name):
        # importing training dataset
        training_samples = pd.read_csv(dataset_name)
        self.data = training_samples.set_index("id_")

        # splitting target column
        self.data_x = self.data.drop(['y'], axis=1)
        self.data_y = self.data['y']

        # training random forest model
        random_forest = RandomForestClassifier(random_state=0)
        random_forest.fit(self.data_x, self.data_y)
        self.ml_model = random_forest

    def get_feature_names(self):
        # getting list of feature names
        return list(self.data_x.columns)
    
    def show_train_dataset(self):
        # showing train dataframe
        display(self.data)