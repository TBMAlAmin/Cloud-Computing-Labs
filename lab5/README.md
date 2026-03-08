# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Dataset
NASA C-MAPSS Turbofan Engine Dataset  
Subset used: FD001

Files:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

## Pipeline

1. Load the FD001 training data
2. Compute Remaining Useful Life (RUL)
3. Extract time-series features using tsfresh
4. Apply filter-based feature selection:
   - Variance Threshold
   - Correlation Filtering
   - Mutual Information
5. Apply Genetic Algorithm (DEAP) for feature selection
6. Train a RandomForestRegressor model
7. Evaluate using RMSE and runtime

## Results

- Samples used: 100
- Selected features: 8
- RMSE: 16.01
- Runtime: 23.27 seconds

## Repository Structure

lab5/
- data/
- components/
  - data_preprocessing/
  - feature_extraction/
  - feature_selection/
  - ga_selection/
  - model_training/
- pipelines/
- README.md

## How to Run

```bash
cd lab5/components/model_training
python train_model.py
