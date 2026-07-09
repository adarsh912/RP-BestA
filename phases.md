# Project Phases Roadmap

This document outlines the **9 Development Phases** for the project: **Adaptive Multi-Feature Linear Fuzzy Information Granulation with Hybrid Similarity Learning for Time Series Classification**.

---

## Phase 1 — Literature Review
* **Goal:** Review time series classification, fuzzy information granulation, and similarity metrics to verify project novelty.
* **Deliverables:**
  - [x] Literature Matrix (35 papers reviewed across 5 categories)
  - [x] Comparison Table of baselines
  - [x] Research Gap Analysis
  - [x] Novelty Report

## Phase 2 — Methodology Design
* **Goal:** Freeze the mathematical architecture, algorithms, and pseudocode.
* **Deliverables:**
  - [x] Mermaid Architecture Block Diagram
  - [x] Formal Mathematical Equations for all steps (Segmentation, Granulation, Features, Similarity, Fusion)
  - [x] Algorithm Pseudocode (Adaptive Granulation, Similarity Fusion)

## Phase 3 — Benchmark Setup
* **Goal:** Setup development environment, requirements, and UCR/UEA loaders.
* **Deliverables:**
  - [x] Python virtual environment (`venv`) & `requirements.txt`
  - [x] Dataset loader (`src/datasets/loader.py`) with pyts & OpenML fallbacks
  - [x] Visualizer and statistics calculator (`src/datasets/stats.py`)

## Phase 4 — Adaptive Segmentation
* **Goal:** Implement and compare fixed, variance, entropy, and change point detection windowing.
* **Deliverables:**
  - [x] Segmentation algorithms (`src/segmentation/adaptive.py`)
  - [x] Visualizer script for segmentation comparisons (`plots/GunPoint_segmentation_comparison.png`)

## Phase 5 — LFIG Construction
* **Goal:** Build linear fuzzy information granules (trends, lower bound, upper bound).
* **Deliverables:**
  - [x] LFIG Granule Builder (`src/granulation/lfig.py`)
  - [x] Visualizer showing envelope spreads on raw data (`plots/GunPoint_lfig_granulation.png`)

## Phase 6 — Multi-Feature Granule Representation
* **Goal:** Extract 10-dimensional statistical/structural feature vectors per granule.
* **Deliverables:**
  - [x] 10D feature extractor (`src/features/extractor.py`)
  - [x] Feature importance Gini rankings with Random Forest (`src/features/importance.py`)

## Phase 7 — Hybrid Similarity Learning
* **Goal:** Implement interval Hausdorff distance, trend DTW, and 10D Cosine DTW with distance/rank fusion.
* **Deliverables:**
  - [x] Pairwise hybrid distance module (`src/similarity/hybrid.py`)
  - [x] Fused weighted distance and Borda count aggregation

## Phase 8 — Classification
* **Goal:** Implement boosting models, kernel SVM, and distance-based kNN classifiers.
* **Deliverables:**
  - [x] Custom Distance kNN & Kernel SVM classifiers (`src/classifiers/models.py`)
  - [x] Boosting integration (XGBoost, LightGBM, CatBoost, RF)

## Phase 9 — Evaluation
* **Goal:** Evaluate the complete pipeline on UCR datasets, compare accuracy, runtimes, and execute ablation studies.
* **Deliverables:**
  - [x] Benchmark comparison script
  - [x] Accuracy/F1/Runtime results table
  - [x] Wilcoxon significance tests
  - [x] Feature ablation analysis

