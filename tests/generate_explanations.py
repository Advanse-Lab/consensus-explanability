import os

from consensus_module.consensus_module import ConsensusModule

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../datasets/")

k_values = [1, 3, 5]

consensus = ConsensusModule()

# Generating explanations for each group of data
for k in k_values:
    print("Generating explanations for the first group")
    consensus.set_samples_dataset(
        samples_dataset_path=datasets_path+"1k_samples_cluster1_85_95.csv", 
        id_column="id_"
    )
    consensus.export_top_k_ranking("Cluster1_85_95_1k_Samples", k)

    print("Generating explanations for the second group")
    consensus.set_samples_dataset(
        samples_dataset_path=datasets_path+"1k_samples_cluster1_95.csv", 
        id_column="id_"
    )
    consensus.export_top_k_ranking("Cluster1_95_1k_Samples", k)

    print("Generating explanations for the third group")
    consensus.set_samples_dataset(
        samples_dataset_path=datasets_path+"1k_samples_cluster2_85_95.csv", 
        id_column="id_"
    )
    consensus.export_top_k_ranking("Cluster2_85_95_1k_Samples", k)

    print("Generating explanations for the forth group")
    consensus.set_samples_dataset(
        samples_dataset_path=datasets_path+"1k_samples_cluster2_95.csv", 
        id_column="id_"
    )
    consensus.export_top_k_ranking("Cluster2_95_1k_Samples", k)