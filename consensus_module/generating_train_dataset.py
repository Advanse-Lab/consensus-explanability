import pandas as pd

# Generating 150k Dataset with random instances from yes and no datasets
path_datasets = "../datasets/"
data_yes = pd.read_csv(path_datasets+"Dataset_99k_yes.csv")
data_no = pd.read_csv(path_datasets+"Dataset_75k_no.csv")

data_yes = data_yes.set_index("id_")
data_no = data_no.rename(columns={"id": "id_"})
data_no = data_no.set_index("id_")

sample_data_yes = data_yes.sample(n=75000)
sample_data_no = data_no.sample(n=75000)

training_samples = pd.concat([sample_data_yes, sample_data_no], axis=0)
training_samples.to_csv(path_datasets+"Random_Generated_Dataset_150k.csv")