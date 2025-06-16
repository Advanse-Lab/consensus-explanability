from consensus_module.internal_explainers.exp_shap import ExpSHAP
from consensus_module.internal_explainers.exp_lime import ExpLIME
from consensus_module.internal_explainers.exp_anchors import ExpAnchors

import json
import warnings
import os

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from consensus_module.utils import get_df_row, get_df_row_values, get_df_feature_names

class InternalExplainers: 
    def __init__(self, ml_model, data_x, shap_file_explainer = "./shap_explainer", jsons_path = "other_approach_jsons"):
        self.feature_names = get_df_feature_names(data_x)
        absolute_path = os.path.dirname(__file__)
        self.json_path = os.path.join(absolute_path, jsons_path)
        
        self.shap_exp = ExpSHAP(ml_model, self.feature_names, shap_file_explainer)
        self.lime_exp = ExpLIME(data_x, ml_model, self.feature_names)
        self.anchors_exp = ExpAnchors(data_x, ml_model, self.feature_names)

    def export_explanations(self, ml_model, samples, indexes, json_name):
        self.ml_model = ml_model
        explanations = dict()
        for row_n in range(0, len(samples)):
            json_output = dict()
            # gives values of ml model predictions
            predict = self.ml_model.predict_proba(samples.iloc[[row_n]])[0]
            json_output['forest_prediction'] = {'not to refactor': predict[0], 'refactor': predict[1]}
            # var to indicate if instance must be refactored or not
            self.prediction_bool = 1 if predict[1] >= 0.5 else 0
            # calls funtions that run and export shap, lime and anchors explanations
            row = get_df_row(samples, row_n)
            row_values = get_df_row_values(samples, row_n)
            
            # print("Generating SHAP explanations...")
            json_output['shap'] = self.shap_exp.export_shap_exp(row, self.prediction_bool)
            # print("Generating LIME explanations...")
            json_output['lime'] = self.lime_exp.export_lime_exp(row, row_values, self.prediction_bool)
            # print("Generating Anchors explanations...")
            json_output['anchors'] = self.anchors_exp.export_anchors_exp(row, row_values)
            # puts explanations in each instance index
            explanations[int(indexes[row_n])] = json_output
        #export explanations to json file
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)
        file_name = f"{self.json_path}/{json_name}_other_top_features_ranking.json"
        with open(file_name, "w") as outfile:
            json.dump(explanations, outfile)
        return explanations