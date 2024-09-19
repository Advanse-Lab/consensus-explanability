import shap
import joblib as jbl

import lime
import lime.lime_tabular

from anchor import anchor_tabular

import numpy as np
import json
import warnings
import os

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from utils import get_df_row, get_df_row_values, get_df_feature_names

# function to help sorting features
def compare_shap_feature_weights(feature_value):
    # highest weights first
    return -feature_value['feature_weight']

def scale_weights(features_weights, features_exp):
    # scale weights to sum to 1
    np_features_weights = np.asarray(features_weights)
    scaled_features_weights = np_features_weights / np_features_weights.sum()
    # add values to dict
    for i in range(0, len(features_exp)):
        features_exp[i]['feature_weight'] = scaled_features_weights[i]
    return features_exp

class InternalExplainers: 
    def __init__(self, ml_model, data_x, shap_file_explainer = "./shap_explainer", jsons_path = "other_approach_jsons"):
        self.feature_names = get_df_feature_names(data_x)
        absolute_path = os.path.dirname(__file__)
        self.json_path = os.path.join(absolute_path, jsons_path)
        
        # Setting up SHAP
        try:
            with open(shap_file_explainer, 'rb') as f:
                self.shap = jbl.load(f)
        except:
            self.shap = shap.TreeExplainer(ml_model)
            print("Gerando shap explainer...")
            with open(shap_file_explainer, 'wb') as f:
                jbl.dump(self.shap, f)

        # Setting up LIME
        self.lime = lime.lime_tabular.LimeTabularExplainer(
            data_x.values,
            feature_names=self.feature_names,
            class_names=[0, 1],
            verbose=True,
            mode='classification', 
            discretize_continuous=True)

        # Setting up Anchors
        self.anchors = anchor_tabular.AnchorTabularExplainer(
            [0, 1],
            self.feature_names,
            data_x.values,
            {})
        
    def run_shap(self, row):
        shap_values = self.shap.shap_values(row)
        return shap_values
    
    def run_lime(self, row_values, ml_model, lime_num_features = 8):
        exp_lime = self.lime.explain_instance(
            row_values,
            ml_model.predict_proba,
            num_features=lime_num_features)
        return exp_lime
    
    def run_anchors(self, row_values, ml_model, anchors_threshold = 0.95):
        exp_anchors = self.anchors.explain_instance(
            row_values,
            ml_model.predict,
            threshold=anchors_threshold)
        return exp_anchors

    # export shap features' explanations
    def export_shap_exp(self, row):
        shap_values = self.run_shap(row)
        
        # get shap values to refactor instance
        shap_output = dict()
        
        features_exp = []
        features_weights = []
        for i in range(0, len(self.feature_names)):
            shap_values_to_refactor = shap_values[i][self.refactor_bool]
            condition = shap_values_to_refactor > 0 if self.refactor_bool else shap_values_to_refactor < 0
            # shap_values has positive values (to refactor) and negative ones (not to refactor)
            if condition:
                f = dict()
                feature_name = self.feature_names[i]
                f['feature_name'] = feature_name
                f['feature_value'] = int(row[feature_name])
                features_weights.append(shap_values_to_refactor)
                f['feature_ranges'] = None
                features_exp.append(f)
        # scale weights to sum to 1
        scaled_features_exp = scale_weights(features_weights, features_exp)
        # sort features by feature weight
        sorted_features_exp = sorted(scaled_features_exp, key=compare_shap_feature_weights)
        # append feature ranking after sort
        rank = 1
        for f in sorted_features_exp:
            f['feature_rank'] = rank
            rank += 1
        shap_output['features'] = sorted_features_exp
        return shap_output

    # export lime features' explanations
    def export_lime_exp(self, row, row_values):        
        exp_lime = self.run_lime(row_values, self.ml_model)
        
        lime_output = dict()
        # general instance indices
        lime_output['intercept'] = exp_lime.intercept[1]
        lime_output['local_prediction'] = exp_lime.local_pred[0]
        lime_features = exp_lime.as_list()
        # features' values
        features_exp = []
        features_weights = []
        rank = 1
        for value in lime_features:
            condition = value[1] > 0 if self.refactor_bool else value[1] < 0
            # value[1] (feature_weight) has positive values (to refactor) and negative ones (not to refactor)
            if condition:
                f = dict()
                # extract feature name from feature ranges string
                any((feature_name := substring) in value[0] for substring in self.feature_names)
                f['feature_name'] = feature_name
                f['feature_value'] = int(row[feature_name])
                features_weights.append(value[1])
                f['feature_ranges'] = value[0]
                f['feature_rank'] = rank # feature's order of priority in explainer's result
                features_exp.append(f)
                rank += 1
        # scale weights to sum to 1
        scaled_features_exp = scale_weights(features_weights, features_exp)
        lime_output['features'] = scaled_features_exp
        return lime_output

    # export anchors features' explanations
    def export_anchors_exp(self, row, row_values):
        anchors_exp = self.run_anchors(row_values, self.ml_model)
        
        anchors_output = dict()
        # general instance indices
        anchors_output['precision'] = anchors_exp.precision()
        anchors_output['coverage'] = anchors_exp.coverage()
        # features' values
        features_exp = []
        features_weights = []
        rank = 1
        for i in range(0, len(anchors_exp.names())):
            f = dict()
            # extract feature name from anchors' names string
            any((feature_name := substring) in str(anchors_exp.names()[i]) for substring in self.feature_names)
            f['feature_name'] = feature_name
            f['feature_value'] = int(row[feature_name])
            weight = (anchors_exp.precision(i)*anchors_exp.coverage(i))/anchors_exp.precision()
            features_weights.append(weight)
            f['feature_weight'] = None
            f['feature_ranges'] = anchors_exp.names()[i]
            f['feature_rank'] = rank # feature's order of priority in explainer's result
            features_exp.append(f)
            rank += 1
        # scale weights to sum to 1
        scaled_features_exp = scale_weights(features_weights, features_exp)
        anchors_output['features'] = features_exp
        return anchors_output

    def export_explanations(self, ml_model, samples, indexes, json_name):
        self.ml_model = ml_model
        explanations = dict()
        for row_n in range(0, len(samples)):
            json_output = dict()
            # gives values of ml model predictions
            predict = self.ml_model.predict_proba(samples.iloc[[row_n]])[0]
            json_output['forest_prediction'] = {'not to refactor': predict[0], 'refactor': predict[1]}
            # var to indicate if instance must be refactored or not
            self.refactor_bool = 1 if predict[1] >= 0.5 else 0
            # calls funtions that run and export shap, lime and anchors explanations
            row = get_df_row(samples, row_n)
            row_values = get_df_row_values(samples, row_n)
            json_output['shap'] = self.export_shap_exp(row)
            json_output['lime'] = self.export_lime_exp(row, row_values)
            json_output['anchors'] = self.export_anchors_exp(row, row_values)
            # puts explanations in each instance index
            explanations[int(indexes[row_n])] = json_output
        #export explanations to json file
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)
        file_name = f"{self.json_path}/{json_name}_other_top_features_ranking.json"
        with open(file_name, "w") as outfile:
            json.dump(explanations, outfile)
        return explanations