# PG-Sequence Learning Surrogate Modeling for Crop Growth and Yield Prediction

## Intro
This repository contains the code needed to reproduce all the experiments in the paper: **Physics-Guided Sequence Learning for Surrogate Modeling: An Application for Crop Growth and Yield Prediction**

## Experiments
Each experiment has two corresponding scripts: one (numbered) that corresponds to the training cycle and one (which is a Jupyter notebook called ```observing_*_.ipynb```) that allows the evaluation and analysis of the trained model. Baseline experiments are indicated by the BL prefix.

## Data
The available data is the one used for training neural models, i.e., the dataset constructed through WOFOST executions. To ensure data anonymity, categorical variables that could reveal identity are transformed into ordinal variables. Similarly, the coordinates of the plots are shifted in latitude and longitude by an unknown amount.
