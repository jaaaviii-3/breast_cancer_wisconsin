# Breast Cancer Wisconsin: Model Comparison for Tumor Classification

This repository contains a machine learning project focused on the classification of breast cancer tumors as **benign (B)** or **malignant (M)** using the **Breast Cancer Wisconsin Diagnostic Dataset** from the UCI Machine Learning Repository.

The objective of the project is to compare several classification models using different performance metrics and determine which model is the most appropriate for this dataset, with special attention to the correct classification of malignant tumors.

## Project Overview

The dataset contains 569 observations and 32 variables:

* 1 identifier variable: `ID`
* 1 target variable: `Diagnosis`
* 30 numerical predictor variables describing tumor characteristics

Each tumor is described using 10 morphological characteristics, each measured in three different ways:

* `mean`
* `se` (standard error)
* `worst`

The ten base characteristics are:

* radius
* texture
* perimeter
* area
* smoothness
* compactness
* concavity
* concave points
* symmetry
* fractal dimension

The target variable is:

* `B`: benign tumor
* `M`: malignant tumor

The dataset contains approximately 37.26% malignant tumors and 62.74% benign tumors, so there is no severe class imbalance.

## Models Compared

The following models were trained and evaluated:

1. Linear Discriminant Analysis (LDA)
2. Principal Component Analysis + Linear Discriminant Analysis (PCA + LDA)
3. Classification Tree
4. Pruned Classification Tree
5. Random Forest
6. Multilayer Neural Network (MLP)

The models were mainly implemented in **RStudio**, except for the neural network, which was implemented in **Python** using **TensorFlow/Keras**.

## Evaluation Metrics

The models were compared using the following metrics:

* **Accuracy**: proportion of correctly classified tumors.
* **Sensitivity**: proportion of benign tumors correctly classified.
* **Specificity**: proportion of malignant tumors correctly classified.

In this project, malignant tumors are the most critical class from a decision-making perspective. Therefore, special attention is paid to the number of malignant tumors incorrectly classified as benign.

A cost-sensitive evaluation was also considered, assigning a higher penalty to false negatives than to false positives.

## Results Summary

| Model                   | Accuracy (%) | Sensitivity (%) | Specificity (%) |
| ----------------------- | -----------: | --------------: | --------------: |
| LDA                     |        95.61 |           98.36 |           92.45 |
| PCA + LDA 3 components  |        93.86 |          100.00 |           86.79 |
| PCA + LDA 5 components  |        94.74 |          100.00 |           88.68 |
| PCA + LDA 7 components  |        94.74 |          100.00 |           88.68 |
| PCA + LDA 14 components |        94.74 |          100.00 |           88.68 |
| Classification Tree     |        92.98 |           93.44 |           92.45 |
| Random Forest           |        94.74 |           95.08 |           94.34 |
| Neural Network          |       100.00 |          100.00 |          100.00 |

The neural network achieved perfect classification on the selected test partition. However, this result should be interpreted with caution, since it corresponds to a specific train/test split and may not necessarily generalize to other partitions or external datasets.

## Main Findings

The exploratory analysis showed strong positive linear correlations among many predictor variables. This is expected because the dataset contains different measurements of related tumor characteristics, such as radius, perimeter and area.

PCA was tested as a dimensionality reduction method before applying LDA. However, although PCA reduced the number of input variables, it did not improve the classification performance compared to the direct LDA model.

The classification tree provided a more interpretable model, but its predictive performance was lower than that of Random Forest and the neural network.

Random Forest achieved strong results and provided a robust benchmark, with high classification performance and a low number of critical errors.

The multilayer neural network achieved the best result on the selected test set, correctly classifying all benign and malignant tumors. Nevertheless, this result should be validated with repeated splits or cross-validation before drawing a definitive conclusion.

## Technologies Used

* R
* RStudio
* Python
* VSCode
* TensorFlow/Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

## Dataset Source

The dataset used in this project is the Breast Cancer Wisconsin Diagnostic Dataset from the UCI Machine Learning Repository.

Dataset URL:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

## Limitations

This project is intended as a machine learning model comparison exercise and should not be interpreted as a clinically validated diagnostic tool.

Main limitations include:

* The dataset is relatively small.
* The results depend on the selected train/test partition.
* No external validation dataset was used.
* The neural network result should be validated using repeated cross-validation.
* The analysis is based on extracted numerical features, not raw medical images.

## Future Work

Possible improvements include:

* Repeating the experiment using cross-validation.
* Testing additional models such as logistic regression, SVM or gradient boosting.
* Performing threshold optimization to reduce false negatives.
* Adding ROC-AUC and precision-recall analysis.
* Studying model interpretability using SHAP or feature importance methods.
* Validating the models on an external dataset.

## Conclusion

The project shows that several machine learning models can achieve high performance on the Breast Cancer Wisconsin dataset. Among the models tested, the neural network achieved the best performance on the selected test partition, followed by Random Forest and the classification tree.

However, due to the small size of the dataset and the strong performance obtained on a single test split, the neural network result should be interpreted as a strong preliminary result rather than a definitive conclusion. Further validation would be necessary before selecting a final model for any real-world application.
