# DSAI3202 – Lab 4 (Part 1): Text Feature Engineering with Azure ML

**Student:** TBM (60300943)  
**Course:** Cloud Computing – Winter 2026  

---

## Purpose of this lab

The goal of this lab is **not just to generate features**, but to understand how **Azure ML pipelines** allow us to build a **modular, reproducible, and scalable text feature engineering workflow**.

Instead of running scripts manually, each step is isolated as a **command component**, then connected together in a single pipeline.

---

## What I built (and why)

I started with a sampled Amazon Electronics review dataset and built a pipeline that prepares the text for machine learning.

The pipeline performs the following steps **in order**, where each step depends on the previous one:

1. **Split Dataset**  
   The raw dataset is split into train, validation, and test sets to avoid data leakage.

2. **Normalize Text**  
   Review text is cleaned and normalized so that all downstream features are consistent.

3. **Feature Extraction**  
   Multiple types of features are generated from the same normalized text:
   - Text length features (simple statistical signals)
   - Sentiment features (polarity-based signals)
   - TF-IDF features (sparse lexical representation)
   - SBERT embeddings (dense semantic representation)

   Each feature type captures *different information* from the same text.

4. **Merge Features**  
   All extracted features are merged into a single dataset so they can be used together by future models.

---

## Why Azure ML pipelines

Using Azure ML pipelines ensures that:
- Every step is **reproducible**
- Components can be **reused or replaced independently**
- The full workflow can be executed with **one command**
- Failures are isolated to specific steps, making debugging easier

This is much closer to how real-world ML systems are built compared to running scripts locally.

---

## Final output

The final result of the pipeline is a single merged feature file: all_features.parquet
This file contains **all engineered features** and is stored in Azure ML blob storage.

---

## What this lab demonstrates

- Understanding of Azure ML command components
- Correct use of pipeline dependencies
- Practical text feature engineering in the cloud
- Separation of data processing logic into clean, modular steps
