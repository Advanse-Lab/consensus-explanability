import os
from consensus_module import ConsensusModule
from internal_explainers import InternalExplainers
from ml_model import MlModel

absolute_path = os.path.dirname(__file__)

datasets_path = os.path.join(absolute_path, "../datasets/")

rf_model = MlModel(datasets_path+"Random_Generated_Dataset_150k.csv")

shap_file_explainer = os.path.join(absolute_path, "../shap_explainer")
explainers_instance = InternalExplainers(rf_model.ml_model, rf_model.data_x, shap_file_explainer)


# c1_g3_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster0.csv", "id_")
# c1_g3_1k_samples.export_top_k_ranking("Cluster1_Group3_1k_Samples")

# c2_g3_1k_samples = ConsensusModule(datasets_path+"1k_samples_cluster0.csv", "id_")
# c2_g3_1k_samples.export_top_k_ranking("Cluster2_Group3_1k_Samples")

# positive_samples = ConsensusModule(datasets_path+"Amostra-Dataset.csv", "id_")
# positive_samples.export_top_k_ranking("10_positive_samples")

# negative_samples = ConsensusModule(datasets_path+"NO_instances.csv", "id")
# negative_samples.export_top_k_ranking("10_negative_samples")

# yes_samples = ConsensusModule(datasets_path+"Yes_Instances_Experiment.csv", "id", "y")
# yes_samples.export_top_k_ranking("yes_experiment")

# no_samples = ConsensusModule(datasets_path+"No_Instances_Experiment.csv", "id", "Y")
# no_samples.export_top_k_ranking("no_experiment")