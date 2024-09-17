import os
from consensus_module import ConsensusModule

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../case_study_samples/")

c1_85_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster1_85_95.csv", "id_")
c1_85_95_1k_samples.export_top_k_ranking("Cluster1_85_95_1k_Samples", 1)

c1_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster1_95.csv", "id_")
c1_95_1k_samples.export_top_k_ranking("Cluster1_95_1k_Samples", 1)

c2_85_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster2_85_95.csv", "id_")
c2_85_95_1k_samples.export_top_k_ranking("Cluster2_85_95_1k_Samples", 1)

c2_95_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster2_95.csv", "id_")
c2_95_1k_samples.export_top_k_ranking("Cluster2_95_1k_Samples", 1)

# positive_samples = ConsensusModule(datasets_path+"Amostra-Dataset.csv", "id_")
# positive_samples.export_top_k_ranking("10_positive_samples")

# negative_samples = ConsensusModule(datasets_path+"NO_instances.csv", "id")
# negative_samples.export_top_k_ranking("10_negative_samples")

# yes_samples = ConsensusModule(datasets_path+"Yes_Instances_Experiment.csv", "id", "y")
# yes_samples.export_top_k_ranking("yes_experiment")

# no_samples = ConsensusModule(datasets_path+"No_Instances_Experiment.csv", "id", "Y")
# no_samples.export_top_k_ranking("no_experiment")