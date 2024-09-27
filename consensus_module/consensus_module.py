import os

from utils import get_formatted_dataset_and_indexes
from ml_model import MlModel
from internal_explainers import InternalExplainers
from our_approach import OurApproach
from plots import Plot

absolute_path = os.path.dirname(__file__)

class ConsensusModule:
    priority_order = {0: {'explainer': 'rank_anchors', 'explainer_name': 'anchors', 'priority_weight': 3},
                    1: {'explainer': 'rank_shap', 'explainer_name': 'shap', 'priority_weight': 2},
                    2: {'explainer': 'rank_lime', 'explainer_name': 'lime', 'priority_weight': 1}}
    path_datasets = os.path.join(absolute_path, "../datasets/")
    rf_model = MlModel(path_datasets+"Random_Generated_Dataset_150k.csv")

    shap_file_explainer = os.path.join(absolute_path, "../shap_explainer")
    explainers_instance = InternalExplainers(rf_model.ml_model, rf_model.data_x, shap_file_explainer)
    
    def __init__(self, samples_dataset_path, id_column, target_column = None, train_dataset_path = None):
        self.samples, self.samples_indexes = get_formatted_dataset_and_indexes(
            samples_dataset_path,
            id_column,
            target_column)
        
        if train_dataset_path: self.rf_model = MlModel(train_dataset_path)
    
    def export_top_k_ranking(self, samples_name, k = 5, level_of_strictness = 2, poexp = None):
        if k: self.k = k
        if poexp: self.priority_order = poexp
        if level_of_strictness: self.level_of_strictness = level_of_strictness
        
        other_explanations = self.explainers_instance.export_explanations(
            self.rf_model.ml_model,
            self.samples,
            self.samples_indexes,
            samples_name)
        
        our_approach_instance = OurApproach(
            self.rf_model.get_feature_names(),
            self.k,
            self.priority_order,
            self.level_of_strictness,
            samples_name)
        
        our_approach_instance.combine_top_k_features(other_explanations)
        
        all_top_k_rankings = our_approach_instance.generate_top_k_ranking_for_each_approach(other_explanations)
        
        pdf_name = f"{samples_name}_top_{str(self.k)}_rankings.pdf"
        
        Plot(all_top_k_rankings, pdf_name)