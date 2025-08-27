import os

from consensus_module.utils import get_formatted_dataset_and_indexes
from consensus_module.ml_model import MlModel
from consensus_module.internal_explainers.build_internal_explainers import InternalExplainers
from consensus_module.our_approach import OurApproach
from consensus_module.generate_plots import Plot

current_path = os.path.join(os.getcwd())

class ConsensusModule:
    priority_order = {0: {'explainer': 'rank_anchors', 'explainer_name': 'anchors', 'priority_weight': 3},
                    1: {'explainer': 'rank_shap', 'explainer_name': 'shap', 'priority_weight': 2},
                    2: {'explainer': 'rank_lime', 'explainer_name': 'lime', 'priority_weight': 1}}
    
    def __init__(self, train_dataset_path = None, id_column = None, target_column = None, exception = None, model_type = "RF"):
        if train_dataset_path:
            self.ml_model = MlModel(train_dataset_path, id_column, target_column, exception)
        else:
            path_datasets = os.path.join(current_path, "datasets/")
            self.ml_model = MlModel(path_datasets+"Random_Generated_Dataset_150k.csv", 'id_', 'y')

        self.model_type = model_type
        self.ml_model.build_model(self.model_type)

        self.shap_file_explainer = os.path.join(current_path, "./consensus_module/internal_explainers/shap_explainer")
        self.explainers_instance = InternalExplainers(self.ml_model.getMlModel(), self.ml_model.getXData(), self.shap_file_explainer)

    def set_samples_dataset(self, samples_dataset_path, id_column = None, target_column = None):
        if samples_dataset_path != None:
            self.samples, self.samples_indexes = get_formatted_dataset_and_indexes(
                samples_dataset_path,
                id_column,
                target_column)
        else:
            self.samples = self.ml_model.test_x
            self.samples_indexes = self.samples.index.to_list()
    
    def export_top_k_ranking(self, samples_name, k = 5, level_of_strictness = 2, poexp = None):
        if k: self.k = k
        if poexp: self.priority_order = poexp
        if level_of_strictness: self.level_of_strictness = level_of_strictness
        
        print("Generating SHAP, LIME and Anchors explanations:")
        other_explanations = self.explainers_instance.export_explanations(
            self.ml_model.getMlModel(),
            self.samples,
            self.samples_indexes,
            samples_name)
        
        print("Generating our consensual explanations:")
        our_approach_instance = OurApproach(
            self.ml_model.get_feature_names(),
            self.k,
            self.priority_order,
            self.level_of_strictness,
            samples_name)
        
        our_approach_instance.combine_top_k_features(other_explanations)
        
        all_top_k_rankings = our_approach_instance.generate_top_k_ranking_for_each_approach(other_explanations)
        
        print("Plotting rankings")
        pdf_name = f"{samples_name}_top_{str(self.k)}_rankings.pdf"
        
        Plot(all_top_k_rankings, pdf_name)