import random
import os

from ml_model import MlModel
from utils import get_formatted_dataset_and_indexes

def separate_samples_by_category(model_rf, cluster, samples_indexes, quant_samples, out_name):
    predict_85_95, predict_95 = ([] for i in range(2))
    for i in range(0, len(cluster)):
        predict = model_rf.predict_proba(cluster.iloc[[i]])[0][1]
        if predict >= 0.85 and predict < 0.95:
            predict_85_95.append(samples_indexes[i])
        elif predict >= 0.95:
            predict_95.append(samples_indexes[i])
    # get 5 random samples from each group if there are more than 4 instances in the group
    samples_predict_85_95 = random.sample(predict_85_95, quant_samples) if len(predict_85_95) >= quant_samples else predict_85_95
    samples_predict_95 = random.sample(predict_95, quant_samples) if len(predict_95) >= quant_samples else predict_95
    print("Predictions beetwen 85% and 95%: ", len(predict_85_95), "\nPredictions higher than 95%: ", len(predict_95))
    
    samples_85_95 =  cluster.filter(items=samples_predict_85_95, axis=0)
    samples_95 =  cluster.filter(items=samples_predict_85_95, axis=0)

    samples_85_95.to_csv(out_name+"_85_95.csv")
    samples_95.to_csv(out_name+"_95.csv")
    return [samples_predict_85_95, samples_predict_95]

absolute_path = os.path.dirname(__file__)
path_datasets = os.path.join(absolute_path, "../datasets/")
rf_model = MlModel(path_datasets+"Random_Generated_Dataset_150k.csv")

c0_data, c0_indexes = get_formatted_dataset_and_indexes(path_datasets+"2508_Cluster_00.csv", "id_", "cluster")
c1_data, c1_indexes = get_formatted_dataset_and_indexes(path_datasets+"2508_Cluster_01.csv", "id_", "cluster")

path_samples = os.path.join(absolute_path, "../case_study_samples/")

separate_samples_by_category(rf_model.ml_model, c0_data, c0_indexes, 1000, path_samples+"1k_samples_cluster1")
separate_samples_by_category(rf_model.ml_model, c1_data, c1_indexes, 1000, path_samples+"1k_samples_cluster2")