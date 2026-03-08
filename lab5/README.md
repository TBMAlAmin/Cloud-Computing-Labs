# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Dataset
NASA C-MAPSS Turbofan Engine Dataset  
Subset used: FD001

Files:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

## Pipeline

1. Load dataset and compute Remaining Useful Life (RUL)
2. Extract time-series features using tsfresh
3. Apply filter-based feature selection:
   - Variance Threshold
   - Correlation Filtering
   - Mutual Information
4. Apply Genetic Algorithm (DEAP) for feature selection
5. Train RandomForestRegressor model
6. Evaluate model performance

## Results

Samples used: 100  
Selected features: 8  

RMSE: 16.01  

Runtime: 23.27 seconds

## How to run

```bash
cd lab5/components/model_training
python train_model.py
