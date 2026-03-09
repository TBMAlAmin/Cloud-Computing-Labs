# Cloud Computing Lab 5 – Predictive Maintenance Pipeline

## Overview
In this lab I built a predictive maintenance pipeline using the NASA turbofan engine dataset. The goal was to estimate the Remaining Useful Life (RUL) of aircraft engines using time-series sensor data. The pipeline prepares the dataset, extracts useful features, selects the most relevant ones, trains a regression model, and runs the training job on Azure Machine Learning.

## Project Structure
lab5/
│
├── components/
│   └── model_training/
│       └── train_model.py
│
├── data/
│   └── train_FD001.txt
│
├── screenshots/
│   └── azure_run_output.png
│
├── conda_dependencies.yaml
└── README.md

## Dataset
The dataset contains sensor readings from multiple aircraft engines that operate until failure. Each row includes an engine ID, the cycle number, operational settings, and several sensor measurements.

To create the prediction target I computed the Remaining Useful Life (RUL) as the difference between the maximum cycle of each engine and the current cycle.

RUL = maximum cycle − current cycle

This value represents how many cycles the engine has left before failure.

## Pipeline Steps

### Data Preparation
First I loaded the dataset using pandas and assigned column names to make the data easier to work with. For each engine I created a 30-cycle sliding window of sensor readings and randomly selected one window per engine. This process produced 100 training samples.

### Feature Extraction
I used the tsfresh library to automatically generate statistical features from the time-series sensor data. These features summarize patterns in the signals such as statistical properties, trends, and frequency information.

### Feature Filtering
Because tsfresh generates many features, I reduced them using three filtering steps:
- Variance threshold to remove features with almost no variation
- Correlation filtering to remove highly correlated features (threshold 0.95)
- Mutual information ranking to keep the top 20 most informative features

### Genetic Algorithm
To find the best subset of features I applied a genetic algorithm using the DEAP library. The algorithm evaluates different feature combinations and selects the ones that minimize RMSE while slightly penalizing large feature sets.

Configuration used:
Population size: 20  
Generations: 10  
Crossover probability: 0.5  
Mutation probability: 0.2  

The final result selected 11 features.

### Model Training
The final model used was a Random Forest Regressor configured with random_state=42 and 100 trees. The dataset was split into 80% training and 20% testing. Model performance was evaluated using Root Mean Squared Error (RMSE).

## Azure Machine Learning Execution
The training pipeline was executed in Azure Machine Learning Studio. I created a custom environment using the conda_dependencies.yaml file, built the environment image in Azure, and then submitted a custom training job.

The lab5 project folder was uploaded and the following command was executed:

python components/model_training/train_model.py

The job ran on an Azure CPU compute cluster.

## Results
The Azure job completed successfully and produced the following results:

Samples used: 100  
Selected features: 11  
RMSE: 19.75  
Runtime (seconds): 83.02

## Azure Execution Screenshot
The screenshot below shows the successful Azure job output.

<img width="991" height="629" alt="Screenshot 2026-03-09 at 12 20 56 PM" src="https://github.com/user-attachments/assets/0035e509-115e-43e1-8732-8d5834b5186e" />


## Dependencies
The environment configuration is defined in the file conda_dependencies.yaml.

Main libraries used:
- pandas
- numpy
- scikit-learn
- tsfresh
- deap
- tqdm

## Conclusion
In this lab I implemented a predictive maintenance pipeline using time-series sensor data. The pipeline extracts statistical features, selects useful predictors using a genetic algorithm, trains a regression model, and predicts engine remaining useful life. The complete workflow was successfully executed using Azure Machine Learning.
