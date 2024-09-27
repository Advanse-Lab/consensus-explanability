# Consensus Explanation

Explainability in Machine Learning has gained significant atten-
tion and importance in recent years. Understanding how certain
black-box models generate specific outputs has become increas-
ingly crucial. This challenge is particularly evident in refactoring
recommendations, where developers often distrust the suggestions
made by classifiers due to inadequate explanations of how these
recommendations were generated. Consequently, refactoring rec-
ommendation tools are often seen as unreliable. This paper explores
existing explainability techniques and highlights a known issue
among them: the lack of agreement between explainers. To mit-
igate this disagreement, this paper also proposes an agreement
strategy aimed at balancing the explanations and providing more
reliable outputs for developers.

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
#### Instancing `ConsensusModule(...)`
- `samples_dataset_path`: path to your set of samples in format `.csv`;
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