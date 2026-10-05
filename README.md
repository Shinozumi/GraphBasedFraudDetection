# Graph-Based Financial Fraud Detection System

A machine learning system that detects potentially fraudulent financial transactions by combining **XGBoost** with **graph-based transaction-entity features**.

## Overview

Traditional fraud detection models analyze transactions mainly as individual records. This project additionally captures relationships between transactions through shared entities such as:

- Payment cards
- Addresses
- Email domains
- Devices

A transaction-entity graph is constructed using NetworkX, graph features are extracted, and these features are combined with transaction-level features and XGBoost.

## Architecture

```text
IEEE-CIS Dataset
       |
       v
Preprocessing
       |
       v
Feature Engineering
       |
       +--------------------+
       |                    |
       v                    v
Baseline XGBoost     Transaction-Entity Graph
                            |
                            v
                     Graph Features
                            |
                            v
                  Graph-Enhanced XGBoost
                            |
                            v
              Evaluation / Explainability
                            |
                            v
                    Fraud Prediction
```

## Results

| Metric | Baseline | Graph-Enhanced |
|---|---:|---:|
| Precision | 0.2083 | 0.2118 |
| Recall | 0.7352 | 0.7367 |
| F1-score | 0.3247 | 0.3290 |
| ROC-AUC | 0.9027 | 0.9052 |
| PR-AUC | 0.5109 | 0.5131 |

The graph-enhanced model shows an improvement across the evaluated metrics.

## Graph Features

The model uses graph-derived features including:

- Entity count
- Known entity count
- Shared entity count
- Unique entity types
- Average entity degree
- Maximum entity degree
- Connected component size

The graph contains approximately **487K nodes** and **1.39M edges**.

## Technologies

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- NetworkX
- Matplotlib
- pytest

## Project Structure

```text
GraphBasedFraudDetection/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluation.py
│   ├── graph_builder.py
│   ├── graph_features.py
│   ├── train_baseline.py
│   ├── train_graph_model.py
│   ├── explain_model.py
│   ├── error_analysis.py
│   └── predict.py
│
├── tests/
├── models/
├── reports/
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline model:

```bash
python -m src.train_baseline
```

Build the transaction-entity graph:

```bash
python -m src.build_graph
```

Generate graph features:

```bash
python -m src.create_graph_features
```

Train the graph-enhanced model:

```bash
python -m src.train_graph_model
```

Run a sample prediction:

```bash
python -m src.predict
```

Run tests:

```bash
python -m pytest -q
```

## Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset.

The raw dataset is not included in this repository.

## Key Concepts

- Imbalanced classification
- XGBoost
- Feature engineering
- Graph-based feature engineering
- Bipartite graphs
- Entity relationships
- Data leakage prevention
- Chronological evaluation
- Model explainability
- Error analysis
- Fraud-risk prediction

## Future Improvements

- Graph Neural Networks
- Temporal graph modeling
- Real-time fraud detection
- SHAP-based explanations
- Model drift monitoring
- API deployment
