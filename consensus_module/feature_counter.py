import json
import pandas as pd
import numpy as np
import seaborn as sns
import os
from functools import reduce
import matplotlib.pyplot as plt

def feature_counter(json_exp, feature_names):
    features_counter_dict = dict.fromkeys(feature_names, 0)
    for rank in json_exp:
        instance_rank = json_exp[rank]['top_5_features']
        for feature in instance_rank:
            f = feature["feature_name"]
            features_counter_dict[f] += 1
        # if len(instance_rank) > 4:
        #     f = instance_rank[4]["feature_name"]
        #     features_counter_dict[f] += 1
    
    sorted_features_counter_dict = sorted(features_counter_dict.items(), key=lambda x:x[1], reverse=True)
    return pd.DataFrame.from_dict(dict(sorted_features_counter_dict), orient='index', columns=['quantity'])

def get_feature_name_from_string(string, feature_names):
    if string:
        any((f := substring) in string for substring in feature_names)
        return f

def measure_feature_agreement(tuple, k):
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
            
    return total_equal_features/k
    
def measure_rank_agreement(tuple, k):
    (ex, ey) = tuple
    count_feature = 0

    for i in range(0, k):
        if ex[i] == ey[i]:
            count_feature += 1   
    return count_feature/k

def get_ranking_metrics(our_approach, shap_approach, lime_approach, anchors_approach, feature_names, k):
    our_list, shap_list, lime_list, anchors_list = ([] for i in range(4))
    
    for o in our_approach:
        our_list.append(get_feature_name_from_string(o, feature_names))
    
    for s in shap_approach:
        shap_list.append(get_feature_name_from_string(s, feature_names))
    
    for l in lime_approach:
        lime_list.append(get_feature_name_from_string(l, feature_names))
    
    for a in anchors_approach:
        anchors_list.append(get_feature_name_from_string(a, feature_names))
        
    explanations_combinations = [
        [(our_list, our_list), (our_list, shap_list), (our_list, lime_list), (our_list, anchors_list)],
        [(our_list, shap_list), (shap_list, shap_list), (shap_list, lime_list), (shap_list, anchors_list)],
        [(our_list, lime_list), (shap_list, lime_list), (lime_list, lime_list), (lime_list, anchors_list)],
        [(our_list, anchors_list), (shap_list, anchors_list), (lime_list, anchors_list), (anchors_list, anchors_list)]
    ]
    
    df = pd.DataFrame(explanations_combinations,
        index=['our_approach', 'shap_approach', 'lime_approach', 'anchors_approach'],
        columns=['our_approach', 'shap_approach', 'lime_approach', 'anchors_approach'])
    
    exps_feature_agreement = df.map(lambda x: measure_feature_agreement(x, k))
    exps_rank_agreement = df.map(lambda x: measure_rank_agreement(x, k))
    
    return exps_feature_agreement, exps_rank_agreement

def get_average_features_metrics(df):
    feature_agreement_metrics = df["feature_agreement"].values
    sum_feature_agreement = reduce(lambda a, b: a.add(b, fill_value=0), feature_agreement_metrics)
    avg_feature_agreement = sum_feature_agreement.div(len(feature_agreement_metrics)).round(2)
    
    rank_agreement_metrics = df["rank_agreement"].values
    sum_rank_agreement = reduce(lambda a, b: a.add(b, fill_value=0), rank_agreement_metrics)
    avg_rank_agreement = sum_rank_agreement.div(len(rank_agreement_metrics)).round(2)

    return avg_feature_agreement, avg_rank_agreement

def generate_heatmap(df, plot_name):
    heatmap = sns.heatmap(df, annot=True, cmap="crest")
    plt.savefig(plot_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.clf()

def feature_metrics(json_exp, feature_names, k, plot_name):
    df = pd.DataFrame(index=list(json_exp.keys()), columns=['feature_agreement', 'rank_agreement'])
    
    for rank in json_exp:        
        exps_feature_agreement, exps_rank_agreement = get_ranking_metrics(
            json_exp[rank]['our_approach'],
            json_exp[rank]['shap_approach'],
            json_exp[rank]['lime_approach'],
            json_exp[rank]['anchors_approach'],
            feature_names, k)
        
        df.loc[rank] = pd.Series({'feature_agreement': exps_feature_agreement, 'rank_agreement': exps_rank_agreement})
        
    df_avg_feature_agreement, df_avg_rank_agreement = get_average_features_metrics(df)
    generate_heatmap(df_avg_feature_agreement, plot_name+"_avg_feature_agreement_top_"+str(k)+".pdf")
    generate_heatmap(df_avg_rank_agreement, plot_name+"_avg_rank_agreement_top_"+str(k)+".pdf")
    

# # Open and read the JSON file
absolute_path = os.path.dirname(__file__)
json_path = os.path.join(absolute_path, 'top_k_rankings_jsons/general_top_1_ranking_Cluster1_85_95_1k_Samples.json')
# Open and read the JSON file
with open(json_path, 'r') as file:
    data_c1_85_95 = json.load(file)

with open('./top_k_rankings_jsons/general_top_1_ranking_Cluster1_95_1k_Samples.json', 'r') as file:
    data_c1_95 = json.load(file)
    
with open('./top_k_rankings_jsons/general_top_1_ranking_Cluster2_85_95_1k_Samples.json', 'r') as file:
    data_c2_85_95 = json.load(file)

with open('./top_k_rankings_jsons/general_top_1_ranking_Cluster2_95_1k_Samples.json', 'r') as file:
    data_c2_95 = json.load(file)

absolute_path = os.path.dirname(__file__)
path_datasets = os.path.join(absolute_path, "../datasets/Random_Generated_Dataset_150k.csv")
training_samples = pd.read_csv(path_datasets)
training_samples = training_samples.set_index("id_")
training_samples = training_samples.drop(['y'], axis=1)

feature_names = list(training_samples.columns)

plots_path = os.path.join(absolute_path, "plots/heatmap/")
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

feature_metrics(data_c1_85_95, feature_names, 1, plots_path+"Cluster1_85_95_1k_heatmap")
feature_metrics(data_c1_95, feature_names, 1, plots_path+"Cluster1_95_1k_heatmap")
feature_metrics(data_c2_85_95, feature_names, 1, plots_path+"Cluster2_85_95_1k_heatmap")
feature_metrics(data_c2_95, feature_names, 1, plots_path+"Cluster2_95_1k_heatmap")

# count_features_c1_85_95 = feature_counter(data_c1_85_95, feature_names)
# count_features_c1_95 = feature_counter(data_c1_95, feature_names)
# count_features_c2_85_95 = feature_counter(data_c2_85_95, feature_names)
# count_features_c2_95 = feature_counter(data_c2_95, feature_names)

# count_features_c1_85_95.to_csv(os.path.join(absolute_path, "statistics/features_quantities_top5_c1_85_95.csv"))
# count_features_c1_95.to_csv(os.path.join(absolute_path, "statistics/features_quantities_top5_c1_95.csv"))
# count_features_c2_85_95.to_csv(os.path.join(absolute_path, "statistics/features_quantities_top5_c2_85_95.csv"))
# count_features_c2_95.to_csv(os.path.join(absolute_path, "statistics/features_quantities_top5_c2_95.csv"))