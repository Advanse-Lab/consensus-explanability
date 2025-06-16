import lime
import lime.lime_tabular

from consensus_module.utils import scale_weights

class ExpLIME:
    def __init__(self, data_x, ml_model, feature_names):
        self.feature_names = feature_names
        self.ml_model = ml_model
        # Setting up LIME
        self.lime = lime.lime_tabular.LimeTabularExplainer(
            data_x.values,
            feature_names=self.feature_names,
            class_names=[0, 1],
            verbose=True,
            mode='classification', 
            discretize_continuous=True)
    
    def run_lime(self, row_values, ml_model, lime_num_features = 8):
        exp_lime = self.lime.explain_instance(
            row_values,
            ml_model.predict_proba,
            num_features=lime_num_features)
        return exp_lime
    
    # export lime features' explanations
    def export_lime_exp(self, row, row_values, prediction_bool):        
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
            condition = value[1] > 0 if prediction_bool else value[1] < 0
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