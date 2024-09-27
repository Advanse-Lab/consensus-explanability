import os
from consensus_module import ConsensusModule

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../datasets/")

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