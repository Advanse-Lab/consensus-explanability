import json
import os
from utils import get_round_percentage

def extract_feature_names(json_features):
    extracted_feature_names = []
    for f in json_features:
        extracted_feature_names.append(f['feature_name'])
    return extracted_feature_names

def get_info_by_feature_name(exp_f, feature_name):
    feature_obj = next(x for x in exp_f if x["feature_name"] == feature_name)
    return feature_obj['feature_value'], feature_obj['feature_weight'], feature_obj['feature_ranges'], feature_obj['feature_rank']

class OurApproach:
    def __init__(self, feature_names, num_features, priority_order, level_of_strictness, samples_name, our_jsons_path = "our_approach_jsons", top_k_jsons_path = "top_k_rankings_jsons"):
        self.feature_names = feature_names
        self.k = num_features
        self.priority_order = priority_order
        self.level_of_strictness = level_of_strictness
        self.json_name = samples_name
        
        absolute_path = os.path.dirname(__file__)
        self.our_jsons_path = os.path.join(absolute_path, our_jsons_path)
        self.top_k_jsons_path = os.path.join(absolute_path, top_k_jsons_path)

    def compare_feature_weights(self, feature_value):
        order = self.priority_order
        if order[0]['explainer'] in feature_value and order[1]['explainer'] in feature_value and order[2]['explainer'] in feature_value:
            return feature_value[order[0]['explainer']], feature_value[order[1]['explainer']], feature_value[order[2]['explainer']]
        elif order[0]['explainer'] in feature_value and order[1]['explainer'] in feature_value:
            return feature_value[order[0]['explainer']], feature_value[order[1]['explainer']]
        elif order[0]['explainer'] in feature_value and order[2]['explainer'] in feature_value:
            return feature_value[order[0]['explainer']], feature_value[order[2]['explainer']]
        elif order[1]['explainer'] in feature_value and order[2]['explainer'] in feature_value:
            return feature_value[order[1]['explainer']], feature_value[order[2]['explainer']]

    def combine_feature_explanations(self, instance):
        total_sum_of_poexp_weights = 0
        for p in self.priority_order:
            total_sum_of_poexp_weights = total_sum_of_poexp_weights + self.priority_order[p]['priority_weight']
        
        # get each explainer features
        shap_f = instance['shap']['features']
        lime_f = instance['lime']['features']
        anchors_f = instance['anchors']['features']

        # get feature names in each explanation
        shap_f_names = extract_feature_names(shap_f)
        lime_f_names = extract_feature_names(lime_f)
        anchors_f_names = extract_feature_names(anchors_f)

        # ONE LIST FOR EACH EXPLAINER
        combine_all, combine_shap_lime, combine_shap_anchors, combine_lime_anchors = ([] for i in range(4))
        compiled_combinations = dict()
        for f in self.feature_names:
            feature_summary = dict()
            feature_summary['feature_name'] = f
            
            # verify if feature is in at least one explanation
            if f in anchors_f_names or  f in shap_f_names or f in lime_f_names:
                priority_order_anchors = priority_order_shap = priority_order_lime = 0
                anchors_weight = shap_weight = lime_weight = 0
                # ANCHORS ranking information
                if f in anchors_f_names:
                    priority_order_anchors = next(self.priority_order[x]["priority_weight"] for x in self.priority_order if self.priority_order[x]["explainer"] == "rank_anchors")
                    feature_value, anchors_weight, anchors_ranges, anchors_rank = get_info_by_feature_name(anchors_f, f)
                    # feature value
                    feature_summary['feature_value'] = feature_value
                    # puts in dictionary
                    feature_summary['rank_anchors'] = anchors_rank
                    feature_summary['weight_anchors'] = anchors_weight
                    feature_summary['range_anchors'] = anchors_ranges
                # SHAP ranking information
                if f in shap_f_names:
                    priority_order_shap = next(self.priority_order[x]["priority_weight"] for x in self.priority_order if self.priority_order[x]["explainer"] == "rank_shap")
                    feature_value, shap_weight, _, shap_rank = get_info_by_feature_name(shap_f, f)
                    # feature value
                    feature_summary['feature_value'] = feature_value
                    # puts in dictionary
                    feature_summary['rank_shap'] = shap_rank
                    feature_summary['weight_shap'] = shap_weight
                # LIME ranking information
                if f in lime_f_names:
                    priority_order_lime = next(self.priority_order[x]["priority_weight"] for x in self.priority_order if self.priority_order[x]["explainer"] == "rank_lime")
                    feature_value, lime_weight, lime_ranges, lime_rank = get_info_by_feature_name(lime_f, f)
                    # feature value
                    feature_summary['feature_value'] = feature_value
                    # puts in dictionary
                    feature_summary['rank_lime'] = lime_rank
                    feature_summary['weight_lime'] = lime_weight
                    feature_summary['range_lime'] = lime_ranges
                
                # calculate the average weight and agreement index
                explainers_sum_of_poexp_weights = priority_order_anchors + priority_order_shap + priority_order_lime
                feature_summary['avg_weight'] = (3*anchors_weight+2*shap_weight+lime_weight)/explainers_sum_of_poexp_weights
                feature_summary['agreement_index'] = explainers_sum_of_poexp_weights/total_sum_of_poexp_weights
                
                # append intersection of feature in appropriate list
                if priority_order_anchors and priority_order_shap and priority_order_lime:
                    combine_all.append(feature_summary)
                elif priority_order_shap and priority_order_lime:
                    combine_shap_lime.append(feature_summary)
                elif priority_order_anchors and priority_order_shap:
                    combine_shap_anchors.append(feature_summary)
                elif priority_order_anchors and priority_order_lime:
                    combine_lime_anchors.append(feature_summary)

        # order each list by importance (1st anchors, 2nd shap, 3rd lime)
        compiled_combinations['combine_all'] = sorted(combine_all, key=lambda feature_value: self.compare_feature_weights(feature_value))
        compiled_combinations['combine_shap_lime'] = sorted(combine_shap_lime, key=lambda feature_value: self.compare_feature_weights(feature_value))
        compiled_combinations['combine_shap_anchors'] = sorted(combine_shap_anchors, key=lambda feature_value: self.compare_feature_weights(feature_value))
        compiled_combinations['combine_lime_anchors'] = sorted(combine_lime_anchors, key=lambda feature_value: self.compare_feature_weights(feature_value))
        
        return compiled_combinations
    
    def combine_top_k_features(self, other_exps_json):
        final_explanations = dict()
        # run each instance
        for i in other_exps_json:
            combinations = self.combine_feature_explanations(other_exps_json[i])
            # 1st step of algorithm
            # get combinations between all explainers
            final_k_features = combinations['combine_all'][0:self.k]
            # verify how many features misses to complete the x top features
            missing_features = self.k - len(final_k_features)
            
            if missing_features and self.level_of_strictness > 1:
                # get x reamining features from each of combinations between 2 explainers
                i_shap_lime_f = combinations['combine_shap_lime'][0:missing_features]
                i_shap_anchors_f = combinations['combine_shap_anchors'][0:missing_features]
                i_lime_anchors_f = combinations['combine_lime_anchors'][0:missing_features]
                # concat lists
                remaining_features = i_shap_lime_f+i_shap_anchors_f+i_lime_anchors_f
                # 2st step of algorithm
                # sort concatenated list
                sorted_remaining_features = sorted(remaining_features, key=lambda feature_value: self.compare_feature_weights(feature_value))
                # concat features selected
                final_k_features = final_k_features + sorted_remaining_features[0:missing_features]

            instance_final_exp = dict()
            instance_final_exp['forest_prediction_to_refactor'] = other_exps_json[i]['forest_prediction']['refactor']
            top_n_features_string = 'top_'+str(self.k)+'_features'
            instance_final_exp[top_n_features_string] = final_k_features
            
            final_explanations[i] = instance_final_exp

        #export explanations to json file
        if not os.path.exists(self.our_jsons_path):
            os.makedirs(self.our_jsons_path)
        file_name = f"{self.our_jsons_path}/our_top_{str(self.k)}_ranking_{self.json_name}.json"
        with open(file_name, "w") as outfile:
            json.dump(final_explanations, outfile)
        
        self.our_approach_explanations = final_explanations
            
        return final_explanations
    
    def get_top_k_features_from_approach(self, approach_exp, write_weight, write_agreement_index):
        top_k_approach = [None] * self.k
        k_approach = self.k if len(approach_exp) >= self.k else len(approach_exp)
        for feature in range(0, k_approach):
            exp_f = approach_exp[feature]
            feature_name = exp_f["feature_name"]
            avg_weight = f" ({get_round_percentage(exp_f[write_weight])}%" if write_weight else ""
            agreement_index = f" - {get_round_percentage(exp_f[write_agreement_index])}%)" if write_agreement_index else ")"
            top_k_approach[feature] = feature_name+avg_weight+agreement_index
        return top_k_approach

    def generate_top_k_ranking_for_each_approach(self, other_explanations):
        general_top_k = dict()
        
        for i in self.our_approach_explanations:
            top_k_each_instance = dict()

            top_k_each_instance["forest_pred"] = self.our_approach_explanations[i]["forest_prediction_to_refactor"]
            top_k_each_instance["our_approach"] = self.get_top_k_features_from_approach(self.our_approach_explanations[i][f"top_{self.k}_features"], "avg_weight", "agreement_index")
            top_k_each_instance["shap_approach"] = self.get_top_k_features_from_approach(other_explanations[i]["shap"]["features"], "feature_weight", 0)
            top_k_each_instance["lime_approach"] = self.get_top_k_features_from_approach(other_explanations[i]["lime"]["features"], "feature_weight", 0)
            top_k_each_instance["anchors_approach"] = self.get_top_k_features_from_approach(other_explanations[i]["anchors"]["features"], "feature_weight", 0)
        
            general_top_k[str(i)] = top_k_each_instance
        #export explanations to json file
        if not os.path.exists(self.top_k_jsons_path):
            os.makedirs(self.top_k_jsons_path)
        file_name = f"{self.top_k_jsons_path}/general_top_{str(self.k)}_ranking_{self.json_name}.json"
        with open(file_name, "w") as outfile:
            json.dump(general_top_k, outfile)
        return general_top_k