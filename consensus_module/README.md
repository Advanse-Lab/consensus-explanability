## An Empirical Analysis

The goal of this empirical analysis is to ***compare the agreement level between EBM and our approach with LIME, SHAP and Anchors using the metrics Feature Agreement (FA) and Rank Agreement (RA) in the context of an EM Recomender.*** [[1](#1), [2](#2)]


Note that our consensus method integrates SHAP, LIME, and Anchors; therefore, its agreement level is expected to surpass that of the individual methods. Including them as baselines allows us to verify whether the consensus truly improves consistency, rather than merely echoing a single explainers behavior.

To enable a comparison between pairs of explainers—rather than individual explanations—we compute the average metric values (FA and RA) across all instances. For example, to compare two explainers, E1 and E2, we apply both to the same set of n labeled instances, generating n explanations per explainer. For each instance, we calculate the agreement metric (FA or RA) between the two corresponding explanations. This process yields n agreement values, one per instance. We then average these values to produce a single metric score that reflects the overall agreement between the two explainers. This strategy follows the same methodology adopted by [[1](#1)].

## Setup of the Empirical Analysis

In order to compare Explainers, we developed a Refactoring Recommender capable of recommending whether a specific OO method should undergo an EM refactoring [[3](#3)]. The instance/sample to be given to the Recommender is an Object-Oriented (OO) method, characterized by a set of source code metrics/features. We then employed LIME, SHAP, ANCHORS, EBM and our approach for generating explanations for instances with different domain characteristics. This recommender was created with the Random Forest (RF) algorithm, since it has been recognized by several works as having good accuracy in the refactoring recommendation  [[4](#4), [5](#5), [6](#6), [7](#7)].

### **The Training Dataset**

The training dataset used in our study is a subset of the dataset introduced by [[4](#4)], a comprehensive and widely adopted dataset in machine learning research on refactoring recommendation, containing approximately 2 million instances. From this dataset, we selected only the instances that underwent Extract Method refactoring, as it is one of the most commonly applied and extensively studied refactorings in the literature [[8](#8)]. This filtering resulted in a balanced training set with 150,000 instances, composed of 75,000 negative examples ("do not refactor") and 75,000 positive examples (labeled as "refactor").

In addition, we selected 23 features from the original 58, prioritizing metrics that capture the complexity of object-oriented methods. These include: Lines of Code (LOC), Coupling Between Objects (CBO), Cyclomatic Complexity (CC), etc. which are commonly used indicators in the context of refactoring analysis.

### **The Test Dataset and Clusters**

In parallel with the creation of the training dataset, we constructed the test dataset, consisting of 99,911 instances also extracted from the [[4](#4)] dataset. These instances meet the following criteria:
i) they are entirely distinct from those used in the training set;
ii) all are positive instances, meaning they originally correspond to refactored methods; and
iii) the class labels were intentionally removed to simulate a realistic, unlabeled scenario.

Thus, for the test dataset, we applied the K-means clustering algorithm, which revealed three distinct clusters: the first cluster (C-high) comprises 4,392 instances characterized by high metric values; the second includes 66,051 instances with low metric values; and the third contains 29,468 instances with medium-level metrics.

This outcome is particularly interesting as it suggests that developers apply refactorings for a variety of reasons, not solely based on high values in code metrics. Such a domain-specific characteristic further underscores the importance of explainability. Developers often expect refactoring recommendations to target code snippets with high complexity or coupling metrics [[9](#9)]. When a recommendation is made for, say, a short and simple method, developers may naturally question its validity. In these cases, clear and trustworthy explanations become even more critical to support decision-making.

### **Instances to be Explained**

After training the Random Forest (RF) model and generating our Extract Method recommender, we applied it to all instances within the three identified clusters. Once the model had labeled all instances, we selected a subset of 4,000 instances for explanation analysis. This selection was based on a two-step filtering process.

The first step was to retain only the instances labeled as positive. This decision aligns with the standard behavior of refactoring recommenders, which typically generate suggestions only for cases deemed refactorable. Including negative instances would not be practical, as their high volume could overwhelm the analysis and dilute its relevance. The second step involved classifying the selected instances based on two criteria:
i) the confidence level of the RF model, and
ii) the value range (high or low), corresponding to the clusters previously identified.

The confidence level represents the model's estimated probability for a given prediction. For example, if the model predicts class 1 (i.e., refactor) with a confidence of 90%, this can be interpreted as a strong indication that the instance should indeed be refactored.

After training the RF and generated our Extract Method Recomender, we provided to it all the instances of the three clusters. Having labeled all of them, we decided to select 4.000 instances for analysing their explanations. Therefore, we performed a two-step-filtering. The first was to select only positive instances. This was done because usually Refactoring Recomenders usually create recommendations just for positive cases. It would not make sense to show recommendations for negative cases as the number of them would be too high.

The second step was classifying the instances according to two parameters: i) the RF confidence level (precision) and ii) the range of values (high or low), which are the clusters already mentioned. This confidence level can be understood as the probability of a prediction be 0 or 1. For example, if the output of the model is 1 and the confidence level is 90%, we can can interpret this as the model is confident that the output is correct.

**Table 1**: Number of Instances in Each Category
| | **Classification of the Instances (RF Precision x High/Low Values)** | |
|:---|:---|:---|
|  | **C-low values** | **C-high values** |
| **G1: RF Prec: 85-95%** | 1000 instances | 1000 instances |
| **G2: RF Prec: \>95%** | 1000 instances | 1000 instances |

We chose to include only RF predictions with a confidence level greater than 85%. This decision is again motivated by a domain-specific consideration. In our view, an IDE equipped with a refactoring recommender should generate suggestions only when the model exhibits a reasonably high degree of confidence. Otherwise, it could overwhelm the developer with excessive and potentially unreliable recommendations, which may lead to distraction or mistrust in the tool.

Based on this filtering, we constructed a final dataset consisting of 4,000 instances, for which we generated explanations to support our empirical analysis. These instances were divided into four equally sized categories (1,000 instances each), as shown in Table <a href="#table:instances_per_category" data-reference-type="ref" data-reference="table:instances_per_category">1</a>. The categorization was based on two factors:

* **Model confidence level**: G1: predictions with confidence between 85% and 95%; G2: predictions with confidence above 95%.
* **Cluster characteristics**: C-high: instances belonging to the cluster with high metric values; C-low: instances from the cluster with low metric values.

This stratification allowed us to explore the influence of model confidence and code characteristics on the quality and consistency of the generated explanations.

### **Customizing the Prototype**

Before conducting the analysis, we customized our prototype with the following configuration:

* **Internal Explainers**. We have registered the three well-known state-of-the-art XAI Explainers: SHAP, LIME and Anchor;
* **Weight of the explainers**. We assigned the highest priority to Anchor, given its ability to generate more precise and faithful explanations. Anchor produces "anchors"—conditions that, when satisfied, guarantee the same prediction as the original model, offering a higher degree of reliability [[10](#10)];
* **Strictness Level**. We set the strictness level to *medium*, which involves generating two sets of features: one containing attributes shared by all three explainers (full consensus), and another with attributes shared by at least two explainers (n-1 Explainers);
* **Number of Features in the Consensual Explanation**. We configured the prototype to generate explanations with $K = 1$ and $K = 3$ features. This choice aims to identify the most influential variables in the model’s decision process, producing concise, focused, and informative explanations.

Subsequently, we generated explanations for the 4000 instances and evaluated the (dis) agreement between the explanation methods using the *Feature Agreement* and *Rank Agreement* metrics.

## Results

<object data="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.png" type="application/png">
    <embed src="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.png">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/Advanse-Lab/consensus-explanability/blob/main/assets/heatmaps.png">Download PDF</a>.</p>
    </embed>
</object>

Figure 1 shows sixteen matrices that illustrate the (dis)agreement between various pairs of explanations. These matrices were generated combining the parameters C-low and C-high, groups of RF precision (G1 and G2), number of features (k) and the metrics Feature Agreement (RA) and Rank Agreement (RA). We will refer to each matrix by the number that identifies it, which is located in the upper left corner.

The first 8 matrices, at the top (matrices from 1 to 8), were generated using the FA metric, while the bottom eight (matrices from 9 to 16) utilize the RA metric.
In the matrices, known as heatmaps, lighter colors indicate strong disagreement, while darker colors indicate strong agreement. Because of space limitations, we concentrate our analysis of the most typical scenarios. Each subsection below focuses on a specific scenario.

### Analyzing the FA metric

For Heatmap 1, our approach achieves an average agreement of 0.78 (SHAP: 0.71, LIME: 0.84 and ANCHOR: 0.81) while EBM shows an average agreement of 0.70 (SHAP: 0.68, LIME: 0.69 and ANCHOR: 0.73). In heatmap 2, both our approach and EBM yield the same average agreement values as in heatmap 1. In heatmap 3, our approach shows an average agreement of 0.69 (SHAP: 0.72, LIME: 0.61 and ANCHOR: 0.74) while EBM shows an average agreement of 0.56 (SHAP: 0.66, LIME: 0.54 and ANCHOR: 0.47). Similarly, in heatmap 4, both approaches replicate the results observed in heatmap 3.

For Heatmap 5, our approach achieves an average agreement of 0.49 (SHAP: 0.32, LIME: 0.49 and ANCHOR: 0.65) while EBM shows an average agreement of 0.30 (SHAP: 0.39, LIME: 0.30 and ANCHOR: 0.20). In heatmap 6, both our approach and EBM yield the same average agreement values as in heatmap 5. In heatmap 7, our approach shows an average agreement of 0.64 (SHAP: 0.70, LIME: 0.65 and ANCHOR: 0.59) while EBM shows an average agreement of 0.53 (SHAP: 0.68, LIME: 0.61 and ANCHOR: 0.31). Similarly, in heatmap 8, both approaches replicate the results observed in heatmap 7.

**Table 2**: Summary of the level of agreement and percentage improvement

| Heatmap | Cluster | Our  | EBM  | improvement (%) |
| ------- | ------- | ---- | ---- | --------------- |
| 1       | C-Low   | 0.78 | 0.70 | 11.4            |
| 3       | C-Low   | 0.69 | 0.56 | 23.2            |
| 5       | C-High  | 0.49 | 0.30 | 63.3            |
| 7       | C-High  | 0.64 | 0.53 | 20.7            |

**This analysis highlights that our approach consistently achieved higher agreement values (FA metric) compared to the EBM technique across the analyzed heatmaps**. The percentage improvement ranged from 11.4% to 63.3%, with the most notable difference observed in the C-High cluster (heatmap 5). These results indicate that our method produces more consensual and stable explanations, especially in contexts with greater feature complexity.

The analysis also reveals that there is no significant difference between categories G1 (Precision between 85-95%) and G2 (Precision between >95%) when k is constant. This suggests that, despite variations in the precision of the Random Forest model, the explanation remains consistent. This is evident in heatmap 1 for G1 and heatmap 2 for G2, where our approach shows the same agreement with SHAP with 0.71, LIME with 0.84, and ANCHOR at 0.81. This trend continues in heatmap 3 and 4, which present equal or very similar values. We believe that a precision above 85% already indicates high reliability, resulting in minimal or no variation in the features and their rankings used to generate the explanations.

### Analyzing the metric FA when increasing the K-features

We analyze the effect of increasing the number of features K=1 for k=3 on the agreement, considering FA metric and both clusters (C-Low and C-Hight). For this analysis we will not distinguish between groups since the result is approximately the same, as previously demonstrated there is no significant difference between precision groups.

For scenario 1, we analyse the C-Low, comparing the heatmap 1 and 3. For scenario 2, C-Hight, we compare the heatmaps 5 and 7. Thus, for scenario 1, our approach shows a minimum average reduction agreement of 0.09 that represents 12.28% while the EBM shows an average agreement reduction of 0.14 that represents 20.48%. For scenario 2, our approach shows an average agreement increase of 0.16 that represents 32.88% while the EBM shows an agreement increase of 79.58%. Although the EBM method presents a higher percentage of relative growth, the absolute values achieved by our approach are consistently higher (for K=3 of 0.65 when compared to 0.53 for EBM). This indicates that, despite a smaller relative variation, our method starts from a more solid base and achieves a higher performance in absolute terms. The analysis reveals that our proposal maintains on average a good agreement with SHAPE, LIME and ANCHOR overcoming the EBM technique.

### Analyzing the metric FA when varying from C-Low to C-High

Considering  k =1, we analyze whether there is a difference in agreement when varying between the C-Low and C-High clusters. To do that, we examine the heatmaps 1 and 5.

For the C-Low cluster, heatmap 1, our approach demonstrates a high average agreement of 0.79 (SHAP: 0.71, LIME: 0.84 and ANCHOR: 0.81) whereas EBM shows an average agreement of 0.7 (SHAP: 0.68, LIME: 0.69 and ANCHOR: 0.73). In contrast, for the C-High cluster, heatmap 5, our approach yields an average  agreement of 0.49 (SHAP: 0.32, LIME: 0.49 and ANCHOR: 0.65), while EBM achieves an average agreement of 0.30 (SHAP: 0.39, LIME: 0.30 and ANCHOR: 0.20). This analysis reveals that our approach maintains a higher average agreement than EBM when the clusters vary from C-Low to C-High. Noticeably, in the C-High cluster, the agreement values are lower. We attribute this decline to the increasing complexity of the methods within the C-High cluster. This complexity leads to a greater number of features available for generating explanations, which affects the significance of each individual feature's contribution.

### Analyzing the agreement level considering FR metric.

We analyze the agreement focusing on Feature ranking (FR) metric. In the K=1 scenario, regardless of the group (G1 and G2) and the cluster (C-Low and C-High), we observe that the FR and FA metrics yield identical values. This means that heatmaps 9 and 10 are equal to heatmaps 1 and 2, and heatmaps 13 and 14 are equal to heatmaps 5 and 6. This occurs because only one feature is analyzed, resulting in coinciding agreement and consequently identical ranking. Therefore, the analysis previously conducted for the FA metric also applies to FR in this case. 

When analyzing the heatmaps for the FR metric at K=3 and considering both clusters, without considering the groups since they have no impact on the explanation. We observe that the quadrant formed by the heatmaps 11, 12, 15, and 16 exhibits the lowest agreement compared to the other quadrants. This indicates a higher level of disagreement among the explanations. Furthermore, heatmaps 15 and 16, which correspond to the C-High clusters, show even lower agreement values. These results suggest that the FR metric is sensitive not only to the increase in K but also to the complexity associated with the Cluster type. Nevertheless, our proposal consistently achieves better agreement values than the EBM technique. This is particularly evident in heatmap 15, where our method achieves an average agreement of 0.31 (SHAP: 0.25, LIME: 0.3, ANCHOR: 0.38) while EBM reaches only 0.22.

## References
<a id="1">[1]</a> 
Satyapriya Krishna, Tessa Han, Alex Gu, Steven Wu, Shahin Jabbari, and Himabindu Lakkaraju. 2024. The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective. Transactions on Machine Learning Research (2024).

<a id="2">[2]</a>
Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, and Himabindu Lakkaraju. 2023. Post Hoc Explanations of Language Models Can Improve Language Models. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates, Inc.

<a id="3">[3]</a>
Martin Folwer. 1999. Refactoring: Improving the Design of Existing Programs.(1999). Google Scholar Google Scholar Digital Library Digital Library (1999).

<a id="4">[4]</a>
Mauricio Aniche, Erick Maziero, Rafael Durelli, and Vinicius HS Durelli.2020. The effectiveness of supervised machine learning algorithms in predicting software refactoring. IEEE Transactions on Software Engineering 48, 4 (2020).

<a id="5">[5]</a>
Di Cui, Qiangqiang Wang, Siqi Wang, Jianlei Chi, Jianan Li, Lu Wang, and Qing-shan Li. 2023. REMS: Recommending Extract Method Refactoring Opportunities via Multi-view Representation of Code Property Graph. In 2023 IEEE/ACM 31st International Conference on Program Comprehension (ICPC). IEEE.

<a id="6">[6]</a>
Di Cui, Siqi Wang, Yong Luo, Xingyu Li, Jie Dai, Lu Wang, and Qingshan Li. 2022. RMove: Recommending Move Method Refactoring Opportunities using Structural and Semantic Representations of Code. In 2022 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE.

<a id="7">[7]</a>
Abdullah M Sheneamer. 2020. An automatic advisor for refactoring software clones based on machine learning. IEEE Access 8 (2020).

<a id="8">[8]</a>
Guisella Armijo, Daniel Santibañez, Rafael Durelli, and Valter Camargo. 2024. On the Employment of Machine Learning for Recommending Refactorings: A Systematic Literature Review. In Anais do XXXVIII Simpósio Brasileiro de Engenharia de Software (Curitiba/PR). SBC, Porto Alegre, RS, Brasil.

<a id="9">[9]</a>
Nikolaos Tsantalis, Ameya Ketkar, and Danny Dig. 2020. RefactoringMiner 2.0. IEEE Transactions on Software Engineering 48, 3 (2020).

<a id="10">[10]</a>
Marco Ribeiro, Sameer Singh, and Carlos Guestrin. 2018. Anchors: High-Precision Model-Agnostic Explanations. Proceedings of the AAAI Conference on Artificial Intelligence 32 (04 2018).