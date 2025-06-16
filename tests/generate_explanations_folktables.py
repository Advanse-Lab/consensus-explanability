import os

from consensus_module.consensus_module import ConsensusModule

current_path = os.path.join(os.getcwd())

datasets_path = os.path.join(current_path, "../stage-xai/csv_pus/")

k_values = [1, 3, 5]

consensus = ConsensusModule(train_dataset_path=datasets_path+"psam_pusa.csv", model_type="LR")

# Generating explanations for each group of data
for k in k_values:
    print("Generating explanations for the test dataset")
    consensus.set_samples_dataset(
        samples_dataset_path=datasets_path+"psam_pusb.csv", 
        id_column="id_"
    )
    consensus.export_top_k_ranking("Cluster1_85_95_1k_Samples", k)