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

Our tool is built in Python 3.11.4 and you must have Python installed to run it.

## Installation

For running our approach, you can first create a virtual enviroment, we recommend the [Anaconda](https://www.anaconda.com/):

```sh
conda create -n project_env python=3.6.3 anaconda
conda activate project_env
```

After you can install all the requirements through the file `requirements.txt`:

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