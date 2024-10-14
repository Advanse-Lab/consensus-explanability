# Consensus Explanation
Explainability has gained significant attention in recent years. Un- derstanding the decision-making process of black-box models has become increasingly crucial in many domains, including the refac- torings recommendation domain. Studies have shown that develop- ers often distrust recommendations made by classifiers/tools due to absence of explanations on how the recommendations are build. However, although explanation techniques are available in the lit- erature, a new problem emerges, which is the lack of agreement among explainers. To cope with this, we propose an agreement approach whose output is a explanation that reflects a consensus among explainers. This approach was implemented as a publicly available prototype that allows the configuration of some param- eters making it customizable to different domains/contexts. We conducted an empirical analysis in the context of refactoring rec- ommendations by comparing three state-of-the art explanation methods with our approach. The analysis was performed using a Random Forest model trained with a refactoring-oriented dataset. The results show that our approach presents less disagreement in a pairwise comparison among LIME, SHAP and Anchors.

## The Prototype
The prototype was developed in Python and it can be found in this GitHub repository. It follows the 3 phases described in Figure below and works only by giving it as input 3 parameters. The repository contains all the necessary instructions about what is mandatory and what must be configured. Besides, it gives examples of what can be sent to configure the Consensus Module with the different possible parameters.
For the time being, our prototype does not have a graphical interface, nor is it implemented in the form of a plug-in ready to be attached to an IDE. As a result, it is necessary to pass the parameters as described in this README. To turn this prototype into a usable, ready-to-use tool (plug-in) we still need to develop a module that monitors and extracts metrics from the source code and passes these values on to this prototype.

<object data="../assets/heatmaps.pdf" type="application/pdf">
    <embed src="../assets/heatmaps.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="../assets/internal_perspective_approach.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Requirements

Our tool is built in Python 3.12.5 and you must have Python installed to run it.

## Installation

For running our approach, you can first create a virtual enviroment, we recommend the [Anaconda](https://www.anaconda.com/):

```sh
conda create --name project_env python=3.12.5 anaconda
conda activate project_env
```

After you can install all the requirements by two possible ways.
Through the file `setup.py` with the command:
```sh
pip install .
```

Or directly through the file `requirements.txt`:
```sh
pip install -r requirements.txt
```

## Usage

An example of usage of the Consensual Module:

```sh
from consensus_module import ConsensusModule

samples_module = ConsensusModule(samples_csv_path, "id_column_name", "target_column_name")
samples_module.export_top_k_ranking("SamplesName")
```

## Module Configurable Parameters
#### Instantianting `ConsensusModule(...)`
- `samples_dataset_path`: path to your set of samples in format `.csv`. Although you can send more than 1 instance, in general there will only be 1;
- `id_column`: name of the id column of the samples dataset that was passed;
- `target_column` *(optional)*: name of the target column of the samples dataset that was passed, if the dataset was passed with the target column;
- `train_dataset_path` *(optional)*: path to the Random Forest train dataset in format `.csv`. If no train dataset is given, we train the model with "Random_Generated_Dataset_150k.csv" available in the repository;

#### Calling function `export_top_k_ranking(...)`
- `samples_name`: name that will be used to generate the output files;
- `k` *(optional)*: the number `k` that will be used to select the number of features in the final top-k ranking. The default value is set to `5`;
- `level_of_strictness` *(optional)*: parameter to configure the level of strictness in selecting the features. The default value is set to `2`. Given `N` Explainers.
    > If the level is 1, we will select features that are common in all Explainers.
    If the level is 2, we will select features that are common in `N-1` Explainers.
    If there are more than 3 Explainers, level 3 select features that are common in `N-2` Explainers and so go on...
- `poexp` *(optional)*: indicates the Priority Of Explainers (POExp) dictionary/table. The default value is set to:
    > {0: {'explainer': 'rank_anchors', 'explainer_name': 'anchors', 'priority_weight': 3},
    1: {'explainer': 'rank_shap', 'explainer_name': 'shap', 'priority_weight': 2},
    2: {'explainer': 'rank_lime', 'explainer_name': 'lime', 'priority_weight': 1}}
Indicating the priority: Anchors > SHAP > LIME

## Module Configurable Explainers
The ConsensusModule can be configured with other Explainers with some modifications in the code, because the method is generic to any explanation following the given format:
>   "explainer_name": {
        "explainer_general_metric": value, ...,
        "features": [
            {
            "feature_name": "name",
            "feature_value": value,
            "feature_weight": value | NULL,
            "feature_ranges": "name <= value" | NULL,
            "feature_rank": value
            },
        ...]
    }