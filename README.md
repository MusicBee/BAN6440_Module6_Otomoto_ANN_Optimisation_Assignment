# Otomoto ANN Optimization Assignment

## Project Overview

This project compares three optimization algorithms for an Artificial
Neural Network (ANN) used to predict customer churn and support
marketing segmentation at Otomoto. The objective is to identify a model
that can help prioritize retention campaigns and allocate marketing
resources more effectively.

-   **Dataset:** `teleconnect.csv`
-   **Target variable:** `Churn` (`1` = churn, `0` = non-churn)

## Objectives

-   Recreate and evaluate a baseline ANN for customer churn prediction.
-   Compare Adam, RMSprop, and Stochastic Gradient Descent (SGD).
-   Evaluate models using multiple classification metrics.
-   Recommend a provisional model based on performance and business
    priorities.
-   Discuss challenges, bias, fairness, limitations, and future
    improvements.

## Dataset and Preparation

The documented preparation process includes:

1.  Excluding the customer identifier from predictive features.
2.  Converting `TotalCharges` to numeric.
3.  Imputing missing values.
4.  One-hot encoding categorical variables.
5.  Standardizing numerical variables.
6.  Splitting the data into training (70%), validation (15%), and test
    (15%) sets using stratification and `random_state = 42`.

To avoid data leakage, preprocessing should be fitted on the training
data only and then applied to validation and test data.

## ANN Configuration

-   **Architecture:** two hidden layers with 32 and 16 neurons.
-   **Hidden-layer activation:** ReLU.
-   **Output layer:** one neuron with sigmoid activation.
-   **Loss function:** binary cross-entropy.
-   **Learning rate:** 0.001.
-   **Batch size:** 32.
-   **Maximum epochs:** 100.
-   **Early stopping:** monitor validation loss, patience of 10 epochs,
    restore best weights.
-   **Classification threshold:** 0.50 for the reported comparison.

## Optimizers Compared

  -----------------------------------------------------------------------
  Optimizer                           Reason for inclusion
  ----------------------------------- -----------------------------------
  Adam                                Baseline optimizer using adaptive
                                      estimates of gradient moments.

  RMSprop                             Adaptive optimizer evaluated for
                                      its parameter-update behaviour and
                                      potential training efficiency.

  SGD                                 Conventional gradient-descent
                                      benchmark for comparison with
                                      adaptive optimizers.
  -----------------------------------------------------------------------

## Reported Test Results

Confirm these values against your final experiment output before
submission.

  Metric                 Adam   RMSprop      SGD
  ------------------ -------- --------- --------
  Accuracy             79.66%    80.70%   79.94%
  Precision            67.37%    70.81%   65.75%
  Recall               45.55%    46.62%   51.25%
  F1-score             54.35%    56.22%   57.60%
  ROC-AUC              84.03%    84.49%   84.12%
  PR-AUC               63.06%    63.85%   62.01%
  Epochs completed         22        15      100

### Interpretation

-   **RMSprop** achieved the highest test accuracy, precision, ROC-AUC,
    and PR-AUC, and completed training in the fewest epochs.
-   **SGD** achieved the highest test recall and F1-score.
-   **Adam** remained a competitive baseline and achieved the strongest
    validation accuracy and validation F1-score in the reported results.
-   No optimizer was best on every metric. Interpret the differences
    cautiously because the comparison reflects a single reported run.

## Model-Selection Recommendation

RMSprop is the **provisional candidate** when targeting efficiency and
discrimination are priorities. SGD remains a strong alternative when
identifying a larger proportion of actual churners is the priority. The
final choice should use validation results, business costs, fairness
assessment, and further testing---not test metrics alone.

## Challenges and Limitations

-   Churn is the minority class, so accuracy alone can conceal missed
    churners.
-   Recall was below 52% for all three optimizers at the 0.50 threshold.
-   The comparison used one train-validation-test split and one reported
    run per optimizer.
-   A 0.50 classification threshold may not be optimal for Otomoto's
    campaign objectives.
-   Predictive metrics do not demonstrate that marketing campaigns
    improve retention or profitability.
-   Fairness cannot be established without subgroup-level performance
    evaluation.

## Recommended Next Steps

1.  Repeat each optimizer experiment across multiple random seeds.
2.  Tune the classification threshold using validation data and explicit
    campaign costs.
3.  Investigate class weighting or other appropriate methods to improve
    churn recall; apply any resampling only to training data.
4.  Evaluate precision, recall, and false-positive/false-negative rates
    across relevant customer groups.
5.  Compare the ANN with logistic regression and at least one tree-based
    model.
6.  Run a controlled retention campaign and measure retention lift,
    campaign conversion, cost per retained customer, and incremental
    revenue.
7.  Monitor data quality and model performance if the model is deployed.

## Tools and Technologies

The documented workflow uses Python and a neural-network workflow with
TensorFlow/Keras, plus data preprocessing and evaluation tools such as
pandas and scikit-learn. Confirm package versions and the exact
environment from your implementation before recording them in a
reproducibility manifest.

## Academic References

Retain sources that you actually consulted and used to validate the
report.

-   Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.
    (2002). SMOTE: Synthetic minority over-sampling technique. *Journal
    of Artificial Intelligence Research, 16*, 321--357.
    https://doi.org/10.1613/jair.953
-   Keras. (n.d.). *Optimizers*. https://keras.io/optimizers/
-   Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
    optimization. In *Proceedings of the 3rd International Conference on
    Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980
-   scikit-learn developers. (n.d.). *Metrics*.
    https://scikit-learn.org/stable/api/sklearn.metrics.html

## AI Use Disclosure

ChatGPT was used to assist with report structure, explanations and
comparison of optimizers, interpretation and presentation of supplied
experimental results, recommendations, and visual design. The
experimental metrics should be attributed to the assignment's model
runs, not to ChatGPT. The author is responsible for verifying source
claims and ensuring that the final submission reflects the work actually
performed.

## Reproducibility Note

This README documents the reported configuration and results. It does
not include the dataset or guarantee that the reported scores will
reproduce without the exact source code, dependency versions,
preprocessing pipeline, and random seeds used during each run.
