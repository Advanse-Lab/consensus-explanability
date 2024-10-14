# An Empirical Analysis

The goal of this empirical analysis is to ***is to compare the disagreement level of our approach with LIME, SHAP and Anchors using the metrics Feature Agreement (FA) and Rank Agreement (RA) in the context of an Extract Method Recomender.*** [[18](#18), [19](#19)].

To to able to compare pairs of Explainers, and not pairs of explanations, we calculate the average of the metric values (FA and RA) for all instances. For example, to compare the Explainers E1 and E2, we ask for them explaining the same set of *n* labeled instances, generating n explanations from each Explainer. So, for each metric (FA or RA) and for each particular instance we have two explanations and the metric calculated. This is done for the *n* instances generating *n* metric values. We then calculate the average of this, what result in the metric that is possible to compare the Explainers. This strategy is the same employed by [[18](#18)].

## Setup of the Empirical Analysis

In order to compare Explainers we developed a Refactoring Recommender capable of recommending whether a specific OO method should undergo an Extract Method refactoring [[12](#12)]. The instance/sample to be given to the Recommender is an Object-Oriented (OO) method, characterized by a set of source code metrics/features. We then employed LIME, SHAP, Anchors and our approach for generating explanations for instances with different domaim-specific characteristics. This recommender was created with the Random Forest (RF) algorithm, since it has been recognized by several works as having a good accuracy in the refactoring recommendation context [[3](#3), [9](#9), [10](#10), [34](#34)].

### **The Training Dataset**

The dataset we have created for training is a subset of the [[3](#3)] dataset, which is a very complete dataset (around 2.000.000 instances) and widely used in research involving refactoring recommendations with ML. So, we have selected from this dataset only instances that suffered Extract Method refactoring, as this is the most used and researched refactoring [[5](#5)]. This resulted in a training dataset with 150*k* instances, being 75*k* of negative (labeled as not to be refactored) and 75*k* of positive (labeled for refactoring). Moreover, we consider 23 features from the original 58, prioritizing metrics related to complexity of OO methods, such as LOC (Lines of Code), CBO (Coupling between Obects) and Cyclomatic Complexity (CC).

### **The Test Dataset and Clusters**

In parallel with the creation of the training dataset, we created the test dataset. This dataset was filled with 99.911 instances also taken from the [[3](#3)] dataset. These instances have the following characteristics: i) they are not present in the training dataset; ii) all of them are positive instances and iii) for all of them we have removed the label.

After creating the test dataset, we applied the K-means clustering algorithm that revealed three distinct clusters. The first with 4.392 instances with high values for the features (C-high). A second one with 66.051 instances with low values for the features and a third one with 29.468 instances with medium values. This was quite interesting because revealed that developers apply refactorings for many reasons, not only for code-snippets with high metric values.

This domain-specific characteristic makes explicability still more important, since developers would normally expect refactoring recommendations for code snippets with high values of the metrics [[37](#37)]. When they receive a recommendation for a short method, for example, naturally they will be unsure whether the recommendation is reliable or not, what makes explanation still more important.

### **Instances to be Explained**

After training the RF and generated our Extract Method Recomender, we provided to it all the instances of the three clusters. Having labeled all of them, we decided to select 4.000 instances for analysing their explanations. Therefore, we performed a two-step-filtering. The first was to select only positive instances. This was done because usually Refactoring Recomenders usually create recommendations just for positive cases. It would not make sense to show recommendations for negative cases as the number of them would be too high.

The second step was classifying the instances according to two parameters: i) the RF confidence level (precision) and ii) the range of values (high or low), which are the clusters already mentioned. This confidence level can be understood as the probability of a prediction be 0 or 1. For example, if the output of the model is 1 and the confidence level is 90%, we can can interpret this as the model is confident that the output is correct.

| **Classification of the Instances (RF Precision x High/Low Values)** |  |  |
|:---|:---|:---|
|  | **C-low values** | **C-high values** |
| **G1: RF Prec: 85-95%** | 1000 instances | 1000 instances |
| **G2: RF Prec: \>95%** | 1000 instances | 1000 instances |

Number of Instances in Each Category

We decided to consider only RF predictions having precision higher than 85%. This is is again supported by a domain characteristic. We envisage that an IDE equipped with a Refactor Recomender must only create recommendations when the model has a "good" confidence level. On the contrary, it would appear too much recommendations for the developer, what could disturb her/him.

We then created a final set containing 4.000 instances whose explanations were generated and used in our empirical analysis. We divided them in four categories of 1000 instances each, as can be seen in Table <a href="#table:instances_per_category" data-reference-type="ref" data-reference="table:instances_per_category">1</a>. Two groups related to the precision/confidence level: G1 for precision between 85% and 95% and G2 for precision higher than 95%. And two categories for the clusters: C-high representing the cluster containing instances with high values for the features and C-low representing the instances with low values.

### **Customizing the Prototype**

Before proceeding with the analysis we customized our prototype as follows:

- **Internal Explainers**. We have registered the three well-known state-of-the-art XAI Explainers: SHAP, LIME and Anchor;

- **Weight of the explainers**. Regarding the priority order, we designated Anchor as the highest priority because it provides more accurate explanations by generating "anchors", which guarantee the same prediction as the original model [[28](#28)];

- **Strictness Level**. We define the strictness level as *medium*, meaning we create two sets: one containing attributes common of all three Explainers and another with the attributes that overlap between pairs of models (n-1 Explainers);

- **Number of Features in the Consensual Explanation**. We set K to 1 and 3, as our goal is to identify the variables that capture the most variability in the model, thereby generating brief, concise, and relevant explanations.

Subsequently, we generated explanations for the 4000 instances and evaluated the (dis) agreement between the explanation methods using the *Feature Agreement* and *Rank Agreement* metrics.

## Results

<object data="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.pdf" type="application/pdf">
    <embed src="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.pdf">Download PDF</a>.</p>
    </embed>
</object>

Figure above shows sixteen matrices, that illustrates the (dis)agreement between various pairs of explanations. These matrices were generated combining the parameters C-low and C-high, groups of RF precision (G1 and G2), number of features (k) and the metrics Feature Agreement (RA) and Rank Agreement(RA). We will refer to each matrix by the number that identifies it, which is located in the upper left corner.

The first 8 matrices, at the top (matrices from 1 to 8), were generated using the FA metric, while the bottom eight (matrices from 9 to 16) utilize the RA metric. In the matrices, known as heatmaps, lighter colors indicate strong disagreement, while darker colors indicate strong agreement. Because of space limitations, we concentrate our analysis in some most typical scenarios. Each subsection below focuses on a specific scenario.

### Analysing low and high values considering FA metric with K=1 and G1

Analyzing matrices 1 and 5, notably, our approach demonstrated highest agreement with LIME and ANCHORs in both matrices . For the C-Low cluster (matrix 1), our approach shows an agreement level of 0.84 with LIME and 0.81 with ANCHOR. In the C-High cluster (matrix 5), our approach shows an strong agreement with the same explainers, but the agreement level decrease to 0.49 for LIME and 0.65 for ANCHOR. We attribute the decline in agreement between clusters to the increasing complexity of the methods that compose the C-High Cluster. This complexity result in a greater number of features available for generating explanations, which affects the significance of each features contribution.

### Analysing the increase of K-features in clusters considering FA metric and both groups of precision

We analyze the effect of increasing the number of features K=1 for k=3 on the agreement level in clusters, considering FA metric and both groups of precision. For C-Low scenario, we compare the matrices 1 and 2 with matrices 3 and 4. The analysis reveals that **our approach increases the number of models it agrees with as the number of features rises** for C-Low context. This is evident in the heatmaps 1 and 2, where our approach demonstrated strong agreement with LIME and ANCHOR. When the number of features increases K=3 (heatmaps 3 and 4) our approach maintains strong agreement with LIME and ANCHOR and also shows agreement with SHAP. However, in the C-High scenario, for comparing matrices 5 and 6 with matrices 7 and 8, our approach exhibits a decrease in the number of models with which it agrees, demonstrating better agreement only with ANCHOR of 0.59. Noticeable, the heatmaps considering C-High clusters reveal a lower average of agreement values. For instance, Matrix 5 shows a value of 0.23 as the lowest agreement between SHAP and ANCHORS, while matrix 7 shows 0.33 as the lowest agreement between the same two explanations. Despite this decline in values for C-High cluster, our approach maintains an agreement level with ANCHOR that exceeds 50%.

Another key aspect to highlight is that there is no significant difference between categories G1 (Precision between 85-95%) and G2 (Precision between \>95%). This indicates that, **despite the precision offered by the Random Forest model, explanation does not vary with its value**. This is evident in the heatmap 1 for G1, where our approach shows agreement with LIME at 0.84 and with ANCHOR at 0.81. Similarly, in the heatmap 2 for G2, our approach demonstrates close agreement with LIME at 0.83 and with ANCHOR at 0.82. This trend continues in the heatmap 3 and 4 where our Approach agrees with LIME at 0.61, SHAP at 0.72 and with ANCHOR at 0.74 for Heatmap 3, presenting similar values for Heatmap 4. We believe that precision greater than 85% represents high reliability, resulting in little or no variation in the features and their rankings used to formulate an explanation.

### Analysing the agreement level considering FR metric and both Clusters

We analyze the agreement level focusing on Feature ranking (FR) metric. In the K=1 scenario, regardless of the group (G1 and G2) and the cluster (C-Low and C-High), we observe that the FR and FA metrics yield the same values. This means, the matrices 9 and 10 are equals to matrices 1 and 2, as well as matrices 13 and 14 are equals to matrices 5 and 6. This occurs because only one feature is analyzed, resulting in coinciding agreement and consequently identical ranking. Because of this, the analysis already performed on heatmaps for FA (1, 2, 5 and 6) also applies to FR.

Analyzing the heatmaps for FR metric at K=3 for both clusters, without considering the groups since they have no impact on the explanation. We observe that the quadrant formed by the matrices 11, 12, 15 and 16 presents the lowest values when compared with the other quadrants, indicating a high level of disagreement between the explanations. Furthermore, matrices 15 and 16 for C-High cluster show even lowest agreement values, suggesting that FR is sensitive not only to the increase in K but also to the type of Cluster. However, our proposal presents the best agreement values, standing out above the other explanations. This is evident in the level of agreement that our approach achieves with SHAP 0.43, Lime with 0.44 and Anchor with 0.59 for matrices 11 and 12, as well as 0.3 with LIME and 0.38 with ANCHORS for matrices 15 and 16.

## References
<a id="3">[3]</a> 
Mauricio Aniche, Erick Maziero, Rafael Durelli, and Vinicius HS Durelli. 2020. The effectiveness of supervised machine learning algorithms in predicting software refactoring. IEEE Transactions on Software Engineering 48, 4 (2020).

<a id="5">[5]</a>
Guisella Armijo, Daniel Santibañez, Rafael Durelli, and Valter Camargo. 2024. On the Employment of Machine Learning for Recommending Refactorings: A Systematic Literature Review. In Anais do XXXVIII Simpósio Brasileiro de Engenharia de Software (Curitiba/PR). SBC, Porto Alegre, RS, Brasil.

<a id="9">[9]</a>
Di Cui, Qiangqiang Wang, Siqi Wang, Jianlei Chi, Jianan Li, Lu Wang, and Qing-shan Li. 2023. REMS: Recommending Extract Method Refactoring Opportunities via Multi-view Representation of Code Property Graph. In 2023 IEEE/ACM 31st International Conference on Program Comprehension (ICPC). IEEE.

<a id="10">[10]</a>
Di Cui, Siqi Wang, Yong Luo, Xingyu Li, Jie Dai, Lu Wang, and Qingshan Li. 2022. RMove: Recommending Move Method Refactoring Opportunities using Structural and Semantic Representations of Code. In 2022 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE.

<a id="12">[12]</a>
Martin Folwer. 1999. Refactoring: Improving the Design of Existing Programs.(1999). Google Scholar Google Scholar Digital Library Digital Library (1999).

<a id="18">[18]</a>
Satyapriya Krishna, Tessa Han, Alex Gu, Steven Wu, Shahin Jabbari, and
Himabindu Lakkaraju. 2024. The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective. Transactions on Machine Learning Research (2024).

<a id="19">[19]</a>
Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, and Himabindu Lakkaraju. 2023. Post Hoc Explanations of Language Models Can Improve Language Models. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates, Inc.

<a id="28">[28]</a>
Marco Ribeiro, Sameer Singh, and Carlos Guestrin. 2018. Anchors: High-Precision Model-Agnostic Explanations. Proceedings of the AAAI Conference on Artificial Intelligence 32 (04 2018).

<a id="34">[34]</a>
Abdullah M Sheneamer. 2020. An automatic advisor for refactoring software clones based on machine learning. IEEE Access 8 (2020).