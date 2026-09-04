# Federated Medical Learning

## Overview

Federated Medical Learning is a privacy-oriented machine learning project that demonstrates how multiple healthcare institutions can collaboratively train a medical diagnosis model without directly sharing their local training data.

The project simulates multiple clients, each training a local machine learning model on its own portion of the dataset. The locally trained models are then aggregated by a central server using **Federated Averaging (FedAvg)** to create an improved global model.

The system also demonstrates non-IID data distribution, configurable federated training, model evaluation, communication-cost tracking, and scalability analysis through an interactive dashboard.

---

## Problem Statement

Healthcare institutions often possess valuable patient data that cannot be freely shared due to privacy, security, and regulatory concerns.

Traditional centralized machine learning requires data from different institutions to be collected and stored in a central location. This creates challenges related to:

- Patient data privacy
- Data security
- Institutional data ownership
- Limited access to diverse datasets
- Differences in data distributions between institutions

This project addresses these challenges by simulating a **Federated Learning** environment where clients collaboratively train a shared model while keeping their training data locally.

---

## Objectives

- Implement a basic Federated Learning framework for medical classification.
- Simulate multiple healthcare clients with distributed datasets.
- Implement **Federated Averaging (FedAvg)** for global model aggregation.
- Demonstrate training with **non-IID data** across clients.
- Compare federated learning performance with a centralized baseline.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Measure communication overhead during federated training.
- Analyze the effect of different numbers of clients and training rounds.
- Provide an interactive dashboard for training, evaluation, and prediction.
- Demonstrate the potential of federated learning for privacy-oriented healthcare AI.

---

## Dataset

The project uses the **Breast Cancer Wisconsin dataset** provided through `scikit-learn`.

The dataset contains numerical medical features used to classify cases into two categories:

- **Malignant**
- **Benign**

The dataset contains **569 samples** and **30 numerical features**.

The data is divided into training and testing sets using an 80:20 split, with stratification to maintain class distribution.

Before training, the features are standardized using `StandardScaler`.

The training data is then distributed among multiple simulated clients to represent different healthcare institutions.

---

## Architecture

The system follows a client-server Federated Learning architecture.

```text
                         ┌─────────────────────┐
                         │   Central Server    │
                         │                     │
                         │   Global Model      │
                         │   FedAvg Aggregator │
                         └──────────┬──────────┘
                                    │
                              Global Model
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
      ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
      │   Client 1  │       │   Client 2  │       │   Client N  │
      │             │       │             │       │             │
      │  Local Data │       │  Local Data │       │  Local Data │
      │  Local Model│       │  Local Model│       │  Local Model│
      └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                            Model Updates
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   FedAvg Aggregator │
                         │                     │
                         │ Updated Global Model│
                         └─────────────────────┘
```

### Training Process

1. The medical dataset is loaded and preprocessed.
2. Training data is distributed across multiple simulated clients.
3. The central server sends the current global model to each client.
4. Each client trains the model locally using its own data.
5. Clients return their updated model parameters.
6. The server aggregates the updates using Federated Averaging.
7. The updated global model is evaluated on the test dataset.
8. The process is repeated for multiple communication rounds.

The project uses an **SGD-based logistic classification model** for local training.

---

## Educational Value

This project provides a practical demonstration of the intersection of **Machine Learning, Distributed Computing, Privacy-Preserving AI, and Healthcare**.

It helps demonstrate key concepts including:

- Federated Learning
- Federated Averaging
- Distributed model training
- Client-server architecture
- Non-IID data
- Local model training
- Global model aggregation
- Medical classification
- Model evaluation
- Communication overhead
- Federated learning scalability

The project also provides an understanding of why federated learning can be useful in situations where centralized data collection is difficult or undesirable.

---

## Results

The system evaluates the federated model across multiple communication rounds and compares its performance with a centralized machine learning baseline.

The following metrics are recorded:

| Metric | Description |
|---|---|
| Accuracy | Overall classification performance |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify positive cases |
| F1-Score | Balance between precision and recall |
| Communication Cost | Estimated model-update communication |
| Round Time | Time required for each federated round |

The project also supports experiments with different numbers of clients and training configurations to analyze **model performance, training time, and communication overhead**.

The interactive dashboard provides a visual representation of the training process and allows users to experiment with different federated learning parameters.

---

## Disclaimer

This project is developed for **academic and educational purposes only**.

The system demonstrates federated learning concepts using a public medical dataset and simulated clients. It is **not a clinically validated system** and should not be used for medical diagnosis, treatment decisions, or other clinical applications.

The implementation demonstrates data locality during simulated federated training but does not provide formal privacy guarantees such as differential privacy or secure aggregation.
