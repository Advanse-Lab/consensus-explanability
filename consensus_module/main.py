import os
from consensus_module import ConsensusModule

from external_explainers.interpret_ml_explainer import InterpretMlModel

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../datasets/")

# Generating EBM explanations
k = 1
ebm_explainer = InterpretMlModel()
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
# For k=1
c1_85_95_1k_samples_top1 = ConsensusModule(datasets_path+"1k_samples_cluster1_85_95.csv", "id_")
c1_85_95_1k_samples_top1.export_top_k_ranking("Cluster1_85_95_1k_Samples", 1)

c1_95_1k_samples_top1 = ConsensusModule(datasets_path+"1k_samples_cluster1_95.csv", "id_")
c1_95_1k_samples_top1.export_top_k_ranking("Cluster1_95_1k_Samples", 1)

c2_85_95_1k_samples_top1 = ConsensusModule(datasets_path+"1k_samples_cluster2_85_95.csv", "id_")
c2_85_95_1k_samples_top1.export_top_k_ranking("Cluster2_85_95_1k_Samples", 1)

c2_95_1k_samples_top1 = ConsensusModule(datasets_path+"1k_samples_cluster2_95.csv", "id_")
c2_95_1k_samples_top1.export_top_k_ranking("Cluster2_95_1k_Samples", 1)

# For k=3
c1_85_95_1k_samples_top3 = ConsensusModule(datasets_path+"1k_samples_cluster1_85_95.csv", "id_")
c1_85_95_1k_samples_top3.export_top_k_ranking("Cluster1_85_95_1k_Samples", 3)

c1_95_1k_samples_top3 = ConsensusModule(datasets_path+"1k_samples_cluster1_95.csv", "id_")
c1_95_1k_samples_top3.export_top_k_ranking("Cluster1_95_1k_Samples", 3)

c2_85_95_1k_samples_top3 = ConsensusModule(datasets_path+"1k_samples_cluster2_85_95.csv", "id_")
c2_85_95_1k_samples_top3.export_top_k_ranking("Cluster2_85_95_1k_Samples", 3)

c2_95_1k_samples_top3 = ConsensusModule(datasets_path+"1k_samples_cluster2_95.csv", "id_")
c2_95_1k_samples_top3.export_top_k_ranking("Cluster2_95_1k_Samples", 3)

# For k=5
c1_85_95_1k_samples_top5 = ConsensusModule(datasets_path+"1k_samples_cluster1_85_95.csv", "id_")
c1_85_95_1k_samples_top5.export_top_k_ranking("Cluster1_85_95_1k_Samples", 5)

c1_95_1k_samples_top5 = ConsensusModule(datasets_path+"1k_samples_cluster1_95.csv", "id_")
c1_95_1k_samples_top5.export_top_k_ranking("Cluster1_95_1k_Samples", 5)

c2_85_95_1k_samples_top5 = ConsensusModule(datasets_path+"1k_samples_cluster2_85_95.csv", "id_")
c2_85_95_1k_samples_top5.export_top_k_ranking("Cluster2_85_95_1k_Samples", 5)

c2_95_1k_samples_top5 = ConsensusModule(datasets_path+"1k_samples_cluster2_95.csv", "id_")
c2_95_1k_samples_top5.export_top_k_ranking("Cluster2_95_1k_Samples", 5)
'''