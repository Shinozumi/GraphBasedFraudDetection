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
