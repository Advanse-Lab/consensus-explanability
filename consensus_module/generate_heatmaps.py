import json
import pandas as pd
import numpy as np
import seaborn as sns
import os
from functools import reduce
import matplotlib.pyplot as plt

absolute_path = os.path.dirname(__file__)

class HeatmapGenerator:
    def __init__(self, feature_names, k, plots_path):
        self.feature_names = feature_names
        self.k = k
        self.plots_path = plots_path

    def generate_heatmap(self, our_json_exp, ebm_json_exp, plot_name, plot_title, extention = "pdf"):
        self.our_json_exp = our_json_exp
        self.ebm_json_exp = ebm_json_exp
        self.plot_file_name = plot_name
        self.plot_title = plot_title
        self.file_extention = extention

        df = pd.DataFrame(index=list(self.our_json_exp.keys()), columns=['feature_agreement', 'rank_agreement'])
        
        for rank in self.our_json_exp:   
            exps_feature_agreement, exps_rank_agreement = self.get_ranking_metrics(
                self.our_json_exp[rank]['our_approach'],
                self.our_json_exp[rank]['shap_approach'],
                self.our_json_exp[rank]['lime_approach'],
                self.our_json_exp[rank]['anchors_approach'],
                self.ebm_json_exp[rank])
            
            df.loc[rank] = pd.Series({'feature_agreement': exps_feature_agreement, 'rank_agreement': exps_rank_agreement})
            
        df_avg_feature_agreement, df_avg_rank_agreement = self.get_average_features_metrics(df)

        pdf_path = f"{self.plots_path}Heatmap_{self.plot_file_name}"
        self.save_heatmap(
            df_avg_feature_agreement,
            f"{pdf_path}_avg_feature_agreement_top_{self.k}.{self.file_extention}",
            f"{self.plot_title} - FA (k = {self.k})")
        self.save_heatmap(
            df_avg_rank_agreement,
            f"{pdf_path}_avg_rank_agreement_top_{self.k}.{self.file_extention}",
            f"{self.plot_title} - FR (k = {self.k})")

    def get_ranking_metrics(self, our_approach, shap_approach, lime_approach, anchors_approach, ebm_approach):
        our_list, shap_list, lime_list, anchors_list, ebm_list = ([] for i in range(5))
        
        for o in our_approach:
            our_list.append(self.get_feature_name_from_string(o))
        
        for s in shap_approach:
            shap_list.append(self.get_feature_name_from_string(s))
        
        for l in lime_approach:
            lime_list.append(self.get_feature_name_from_string(l))
        
        for a in anchors_approach:
            anchors_list.append(self.get_feature_name_from_string(a))

        for e in ebm_approach:
            ebm_list.append(e[0])
            
        explanations_combinations = [
            [(our_list, our_list), (our_list, shap_list), (our_list, lime_list), (our_list, anchors_list), (our_list, ebm_list)],
            [(shap_list, our_list), (shap_list, shap_list), (shap_list, lime_list), (shap_list, anchors_list), (shap_list, ebm_list)],
            [(lime_list, our_list), (lime_list, shap_list), (lime_list, lime_list), (lime_list, anchors_list), (lime_list, ebm_list)],
            [(anchors_list, our_list), (anchors_list, shap_list), (anchors_list, lime_list), (anchors_list, anchors_list), (anchors_list, ebm_list)],
            [(ebm_list, our_list), (ebm_list, shap_list), (ebm_list, lime_list), (ebm_list, anchors_list), (ebm_list, ebm_list)]
        ]
        
        df = pd.DataFrame(explanations_combinations,
            index=['our_approach', 'shap_approach', 'lime_approach', 'anchors_approach', 'ebm_approach'],
            columns=['our_approach', 'shap_approach', 'lime_approach', 'anchors_approach', 'ebm_approach'])
        
        exps_feature_agreement = df.map(lambda x: self.measure_feature_agreement(x))
        exps_rank_agreement = df.map(lambda x: self.measure_rank_agreement(x))
        
        return exps_feature_agreement, exps_rank_agreement

    def get_feature_name_from_string(self, string):
        if string:
            any((f := substring) in string for substring in self.feature_names)
            return f

    def measure_feature_agreement(self, tuple):
        (ex, ey) = tuple
        inter_exps = set(ex) & set(ey)
        total_equal_features = len(inter_exps)
        
        # "set" select max 1 None from the rankings, we should treat this problem
        none_count_ex = ex.count(None)
        none_count_ey = ey.count(None)
        # if there is more than 1 None in the ranking we will:
        # a) if we have the same quant of None in both - we should count to the total common features
        if none_count_ex > 1 and none_count_ex == none_count_ey:
            total_equal_features += (none_count_ex-1)
        # b) if we have more than 1 in both, but in diff quant - we get only the None present in both rankings
        # (min of None - the one None that was already counted)
        elif none_count_ex > 1 and none_count_ey > 1 and none_count_ex != none_count_ey:
            total_equal_features += min(none_count_ex, none_count_ey)-1
                
        return total_equal_features/self.k
        
    def measure_rank_agreement(self, tuple):
        (ex, ey) = tuple
        count_feature = 0

        for i in range(0, self.k):
            if ex[i] == ey[i]:
                count_feature += 1   
        return count_feature/self.k

    def get_average_features_metrics(self, df):
        feature_agreement_metrics = df["feature_agreement"].values
        sum_feature_agreement = reduce(lambda a, b: a.add(b, fill_value=0), feature_agreement_metrics)
        avg_feature_agreement = sum_feature_agreement.div(len(feature_agreement_metrics)).round(2)
        
        rank_agreement_metrics = df["rank_agreement"].values
        sum_rank_agreement = reduce(lambda a, b: a.add(b, fill_value=0), rank_agreement_metrics)
        avg_rank_agreement = sum_rank_agreement.div(len(rank_agreement_metrics)).round(2)

        return avg_feature_agreement, avg_rank_agreement

    def save_heatmap(self, df, plot_name, plot_title):
        heatmap = sns.heatmap(df, annot=True, cmap="crest", annot_kws = {"size": '11', "fontweight" : 'bold'})
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=12)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=12)
        
        # Aumentar tamanho do título (se houver)
        plt.title(plot_title, fontsize=14, fontweight='bold')

        plt.savefig(plot_name, format=self.file_extention, dpi=300, bbox_inches='tight')
        plt.clf()

    # Count each feature in all rankings
    def feature_counter(self):
        features_counter_dict = dict.fromkeys(self.feature_names, 0)
        for rank in self.our_json_exp:
            instance_rank = self.our_json_exp[rank][f"top_{self.k}_features"]
            for feature in instance_rank:
                f = feature["feature_name"]
                features_counter_dict[f] += 1
        
        sorted_features_counter_dict = sorted(features_counter_dict.items(), key=lambda x:x[1], reverse=True)

        feature_counter_df = pd.DataFrame.from_dict(dict(sorted_features_counter_dict), orient='index', columns=['quantity'])

        # save in .csv
        stat_path = os.path.join(absolute_path, f"statistics/Features_counter_{self.plot_file_name}_top{self.k}.csv")
        os.makedirs(os.path.dirname(stat_path), exist_ok=True)
        feature_counter_df.to_csv(stat_path)

        return feature_counter_df