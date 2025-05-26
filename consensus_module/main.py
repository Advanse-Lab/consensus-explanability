import os
import pandas as pd
import json

from consensus_module import ConsensusModule

from external_explainers.interpret_ml_explainer import InterpretMlModel
from generate_heatmaps import HeatmapGenerator

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../datasets/")

k_values = [1, 3, 5]

# Generating explanations
'''
for k in k_values:
    c1_85_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster1_85_95.csv", "id_")
    c1_85_95_1k_samples.export_top_k_ranking("Cluster1_85_95_1k_Samples", k)

    c1_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster1_95.csv", "id_")
    c1_95_1k_samples.export_top_k_ranking("Cluster1_95_1k_Samples", k)

    c2_85_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster2_85_95.csv", "id_")
    c2_85_95_1k_samples.export_top_k_ranking("Cluster2_85_95_1k_Samples", k)

    c2_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster2_95.csv", "id_")
    c2_95_1k_samples.export_top_k_ranking("Cluster2_95_1k_Samples", k)
'''
# Generating EBM explanations
'''
ebm_explainer = InterpretMlModel()
for k in k_values: 
    ebm_explainer.predict_local_samples(
        samples_dataset_path=datasets_path+"1k_samples_cluster1_85_95.csv",
        output_name = "c1_85_95",
        id_column = "id_",
        top_k = k
    )

    ebm_explainer.predict_local_samples(
        samples_dataset_path=datasets_path+"1k_samples_cluster1_95.csv",
        output_name = "c1_95",
        id_column = "id_",
        top_k = k
    )

    ebm_explainer.predict_local_samples(
        samples_dataset_path=datasets_path+"1k_samples_cluster2_85_95.csv",
        output_name = "c2_85_95",
        id_column = "id_",
        top_k = k
    )

    ebm_explainer.predict_local_samples(
        samples_dataset_path=datasets_path+"1k_samples_cluster2_95.csv",
        output_name = "c2_95",
        id_column = "id_",
        top_k = k
    )

# ebm_explainer.predict_global()
'''

# Generate heatmap plots

path_datasets = os.path.join(datasets_path, "Random_Generated_Dataset_150k.csv")
training_samples = pd.read_csv(path_datasets)
training_samples = training_samples.set_index("id_")
training_samples = training_samples.drop(['y'], axis=1)
feature_names = list(training_samples.columns)
for k in k_values:
    # Open and read the JSON file
    our_json_path = os.path.join(absolute_path, f'top_k_rankings_jsons/our_approach/top{k}/')
    ebm_json_path = os.path.join(absolute_path, f'top_k_rankings_jsons/ebm/top{k}/')

    with open(f'{our_json_path}general_top_{k}_ranking_Cluster1_85_95_1k_Samples.json') as f: our_data_c1_85_95 = json.load(f)
    with open(f'{our_json_path}general_top_{k}_ranking_Cluster1_95_1k_Samples.json') as f: our_data_c1_95 = json.load(f)
    with open(f'{our_json_path}general_top_{k}_ranking_Cluster2_85_95_1k_Samples.json') as f: our_data_c2_85_95 = json.load(f)
    with open(f'{our_json_path}general_top_{k}_ranking_Cluster2_95_1k_Samples.json') as f: our_data_c2_95 = json.load(f)

    with open(f'{ebm_json_path}c1_85_95_ebm_top{k}_rankings.json') as f: ebm_data_c1_85_95 = json.load(f)
    with open(f'{ebm_json_path}c1_95_ebm_top{k}_rankings.json') as f: ebm_data_c1_95 = json.load(f)
    with open(f'{ebm_json_path}c2_85_95_ebm_top{k}_rankings.json') as f: ebm_data_c2_85_95 = json.load(f)
    with open(f'{ebm_json_path}c2_95_ebm_top{k}_rankings.json') as f: ebm_data_c2_95 = json.load(f)

    plots_path = os.path.join(absolute_path, f"plots/heatmaps/top{k}/")

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    heatmap = HeatmapGenerator(feature_names, k, plots_path)

    heatmap.generate_heatmap(our_data_c1_85_95, ebm_data_c1_85_95, "c1_85_95_1k", "C-low/G1", "png")
    heatmap.generate_heatmap(our_data_c1_95, ebm_data_c1_95, "c1_95_1k", "C-low/G2", "png")
    heatmap.generate_heatmap(our_data_c2_85_95, ebm_data_c2_85_95, "c2_85_95_1k", "C-high/G1", "png")
    heatmap.generate_heatmap(our_data_c2_95, ebm_data_c2_95, "c2_95_1k", "C-high/G2", "png")