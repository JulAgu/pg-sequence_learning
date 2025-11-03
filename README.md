# PG-Sequence Learning Surrogate Modeling for Crop Growth and Yield Prediction
The repository contains the code for reproducing all experiments from the paper: **Physics-Guided Sequence Learning for Surrogate Modeling: An Application for Crop Growth and Yield Prediction**

## Requirements
To run the code from this repository, please ensure that your system meets the following requirements:

- **Software Requirements**: A Python installation into an isolated environment is highly recommended. We used **Python v3.10.0** and miniconda for the environement, but any alternative should be just as good. All the necessary dependencies could be installed directly from the ```requeriments.txt``` file.

- **Hardware Requirements**: Experiments were executed using an NVIDIA RTX A6000 GPU in a machine with 125 Go of RAM. Under this setup the more expensive models take **~24 hours** of training.

## Experiments
Each experiment has two corresponding files: a numbered python script that corresponds to the training cycle and a Jupyter notebook called ```observing_*_.ipynb``` that allows the evaluation and analysis of the trained model. Baseline experiments are indicated by the ```BL``` prefix.

**Each training script generates 2 types of outputs** : **(1)** a text file of the form ```./logs/[experiment_name].txt``` and a **(2)** TensorBoard event directory ```./TF_logs/[experiment_name]/``` which contains information on the evolution of the performance over the training and evaluation sets.

## Data
The dataset was constructed using WOFOST runs on real data from agricultural plots in northwestern France. To ensure data anonymity, categorical variables that could reveal identity are transformed into ordinal variables. Similarly, the coordinates of the plots are shifted in latitude and longitude by an unknown quantity.

### Setting up the repository for exectuting the scripts
Due to github limitations, data should be downloaded from hugging face :

1. Download the dataset from : https://huggingface.co/datasets/JulAgu/PG_S2S-Crop_development
2. Create a ```data/``` directory in the root of the project and **put the dataset inside**.
3. Create a sub-directory ```data/work_data/```, **leave it empty**, it is useful for drawing intermediate elements.
