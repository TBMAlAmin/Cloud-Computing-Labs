# Lab 4 – Text Feature Engineering with Azure ML

## Overview
This lab implements an end-to-end **text feature engineering pipeline** using Azure Machine Learning.  
The goal is to transform raw Amazon Electronics reviews into structured numerical features that can later be used for model training.

The entire workflow is built as a **reproducible Azure ML pipeline** composed of modular components.

---

## Dataset
- **Input dataset:** `amazon_electronics_features_v1_sampled`
- Stored as an Azure ML `uri_folder`
- Contains cleaned and sampled Amazon Electronics reviews

---

## Pipeline Design

The pipeline follows the exact sequence required in the lab instructions:

1. **Split Dataset**
   - Splits the dataset into train / validation / test sets
   - Uses fixed ratios and a seed for reproducibility

2. **Normalize Review Text**
   - Cleans and standardizes review text
   - Applied independently to train, validation, and test splits

3. **Feature Extraction**
   - **Length Features:** character and word length statistics
   - **Sentiment Features:** polarity and subjectivity scores
   - **TF-IDF Features:** sparse bag-of-words representation
   - **SBERT Embeddings:** semantic sentence embeddings

4. **Merge All Features**
   - Combines all extracted features into a single feature dataset
   - Output is stored as a unified feature table for downstream modeling

---

## Azure ML Components
Each step in the pipeline is implemented as a reusable Azure ML **command component**:
- `split_dataset`
- `normalize_text`
- `length_features`
- `sentiment_features`
- `tfidf_features`
- `sbert_embeddings`
- `merge_features`

This modular design makes the pipeline easy to debug, reuse, and extend.

---
## Azure ML Pipeline Execution

The text feature engineering pipeline was successfully executed on Azure Machine Learning.

Azure ML run link:
https://ml.azure.com/experiments/id/acbc0636-5933-49a3-896b-e11e9b538c31/runs/eager_airport_rtzr96pbv3?wsid=/subscriptions/a00dcbea-fd05-4973-82dc-120208b60116/resourceGroups/rg-60300943/providers/Microsoft.MachineLearningServices/workspaces/Amazon-Electronics-Lab-60300943

Screenshots of the pipeline graph and component outputs are provided at the bottom of this README.

## Results
- The pipeline executed successfully end-to-end
- All feature extraction steps completed without errors
- Final merged feature dataset was generated and stored in Azure Blob Storage

Screenshots showing:
- the full pipeline graph
- successful execution of each component  
are provided at the **bottom of this page**.

---

## Screenshots

Screenshots of the completed pipeline and component outputs can be found in the `screenshots/` folder.
