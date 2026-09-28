# Project Overview

## Research question

Does choosing features based on information gain result in a model with accuracy comparable to a model trained on all features for the mushroom edibility classification dataset?

## Objectives

- Rank all 22 original features by information gain with respect to edibility.
- Train logistic regression and decision tree classifiers on all 22 features and on the top 5.
- Compare accuracy, precision, recall, and confusion matrices.
- Compare information-gain ranking with the feature usage of the trained decision tree.

## Experimental setup

- Dataset: UCI Mushroom dataset
- Observations: 8,124
- Original predictors: 22 categorical features
- Split: 75% training / 25% test
- Random state: 42
- Selected features: top 5 by information gain
- Decision tree maximum depth: 5
- Positive class for precision/recall: poisonous

## Main finding

The top-5 feature models achieved nearly the same test-set performance as the full-feature models. The largest accuracy difference was 0.10 percentage point. The information-gain ranking and the decision tree both identified `odor` as the most important feature in the experiment.
