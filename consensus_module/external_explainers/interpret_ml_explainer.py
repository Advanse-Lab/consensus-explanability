import pandas as pd
import os
import joblib
import json

from ml_model import MlModel
from utils import get_formatted_dataset_and_indexes

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
set_visualize_provider(InlineProvider())

absolute_path = os.path.dirname(__file__)

class InterpretMlModel:
    path_datasets = os.path.join(absolute_path, "../../datasets/")
    rf_model = MlModel(path_datasets+"Random_Generated_Dataset_150k.csv")

    ebm_explainer_file = os.path.join(absolute_path, "ebm_explainer.pkl")

    def __init__(self, ebm_explainer_file = absolute_path+"/ebm_explainer.pkl", train_dataset_path = None):
        if train_dataset_path: self.rf_model = MlModel(train_dataset_path)

        # Setting up EBM
        try:
            print("Retrieving EBM explainer...")
            self.ebm = joblib.load(ebm_explainer_file)
            print("Retrieved EBM explainer")
        except:
            self.ebm = ExplainableBoostingClassifier()
            X_train = self.rf_model.getXData()
            y_train = self.rf_model.getYData()
            self.ebm.fit(X_train, y_train)
            print("Generating EBM explainer...")
            joblib.dump(self.ebm, ebm_explainer_file)
        
    def predict_local_samples(self, samples_dataset_path, output_name, id_column, top_k = 5, target_column = None):
        # getting test data and indexes
        x_test, x_test_indexes = get_formatted_dataset_and_indexes(
            samples_dataset_path,
            id_column,
            target_column)

        # predict test samples with random forest
        ml_model = self.rf_model.getMlModel()
        y_test_rf = ml_model.predict(x_test)

        # explain test samples with EBM
        ebm_local = self.ebm.explain_local(x_test, y_test_rf)
        ebm_local_json = ebm_local.data(-1)

        # generate the ranking with the top k features
        top_k_rankings = self.get_top_k_ranking(x_test_indexes, ebm_local_json, top_k)

        # save json file with the ranking
        file_path = f"{absolute_path}/../top_k_rankings_jsons/ebm/top{top_k}/{output_name}_ebm_top{top_k}_rankings.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(top_k_rankings, f, indent=2, default=str)

    def get_top_k_ranking(self, x_test_indexes, ebm_exp, top_k):
        top_k_rankings = dict()

        ebm_exp = ebm_exp['specific']

        for index, exp in enumerate(ebm_exp):
            # get the features, its weights and the ml model prediction
            names = exp["names"]
            weights = exp["scores"]
            predict = exp["perf"]["actual"]

            contributions = []
            for n, w in zip(names, weights):
                contributions.append((n, w))

            # sort the feature weights relative to the prediction
            if predict:
                contributions.sort(key=lambda x: x[1], reverse=True)
            else:
                contributions.sort(key=lambda x: x[1], reverse=False)

            sample_index = x_test_indexes[index]
            # get only the top k features
            top_k_rankings[sample_index] = contributions[:top_k]
        
        return top_k_rankings

    
    def predict_global(self):
        ebm_global = self.ebm.explain_global()

        ebm_global_json = ebm_global.data()

        with open("ebm_global.json", "w") as f:
            json.dump(ebm_global_json, f, indent=2, default=str)