"""
Federated Learning for Privacy-Preserving Medical Diagnosis
=============================================================
Cloud CIA Project - Group 10

Simulates a multi-cloud federated learning framework where several
hospital/cloud sites each hold private patient data. Only model
parameters (never raw patient records) are exchanged with a central
aggregation server, using the FedAvg algorithm.

Dataset: Breast Cancer Wisconsin (Diagnostic) - binary classification
(malignant / benign), a standard stand-in for real patient diagnostic data.

Objectives addressed:
1. Secure federated framework for multi-cloud data sharing   -> FedAvg loop
2. Privacy: raw data never leaves its local client            -> only weights shared
3. Reduced communication overhead / resource utilization      -> tracked & plotted
4. Evaluation: security, accuracy, scalability, performance    -> metrics + scalability test
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
import json

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Data loading & simulated multi-cloud partitioning
# ---------------------------------------------------------------------------
def load_and_partition_data(num_clients=4, non_iid=True, test_size=0.2):
    """
    Loads the medical diagnosis dataset and splits it across `num_clients`
    simulated hospital/cloud sites. Each site's data never leaves that site
    during training -- this function only exists to SET UP the simulation.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if non_iid:
        # Sort by label to create skewed ("non-IID") client shards, which is
        # realistic: different hospitals see different patient populations.
        order = np.argsort(y_train)
        X_train, y_train = X_train[order], y_train[order]
    else:
        perm = np.random.permutation(len(y_train))
        X_train, y_train = X_train[perm], y_train[perm]

    client_X = np.array_split(X_train, num_clients)
    client_y = np.array_split(y_train, num_clients)

    return client_X, client_y, X_test, y_test, data.feature_names


# ---------------------------------------------------------------------------
# 2. Local client training
# ---------------------------------------------------------------------------
def local_train(X, y, coef, intercept, classes, local_epochs=1, lr=0.01):
    """
    One 'hospital cloud' trains a local logistic-regression model,
    starting from the current global weights, on ITS OWN private data only.
    Returns the updated local weights (never the data itself).
    """
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=lr,
        random_state=RANDOM_SEED,
        warm_start=True,
    )
    # Seed the model with the current global parameters
    model.classes_ = classes
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()

    for _ in range(local_epochs):
        model.partial_fit(X, y, classes=classes)

    return model.coef_, model.intercept_, len(y)


# ---------------------------------------------------------------------------
# 3. Server-side aggregation (FedAvg)
# ---------------------------------------------------------------------------
def federated_average(client_updates):
    """
    Weighted average of client model parameters, weighted by each client's
    local dataset size. This is the ONLY thing that crosses the network.
    """
    total_samples = sum(n for _, _, n in client_updates)
    new_coef = sum(coef * n for coef, _, n in client_updates) / total_samples
    new_intercept = sum(icpt * n for _, icpt, n in client_updates) / total_samples
    return new_coef, new_intercept


# ---------------------------------------------------------------------------
# 4. Full federated training loop
# ---------------------------------------------------------------------------
def run_federated_learning(num_clients=4, rounds=15, local_epochs=1, non_iid=True, verbose=True):
    client_X, client_y, X_test, y_test, feature_names = load_and_partition_data(
        num_clients=num_clients, non_iid=non_iid
    )
    n_features = client_X[0].shape[1]
    classes = np.array([0, 1])

    # Global model starts at zero
    global_coef = np.zeros((1, n_features))
    global_intercept = np.zeros(1)

    history = {"round": [], "accuracy": [], "f1": [], "comm_bytes": [], "round_time_s": []}
    bytes_per_update = (n_features + 1) * 8  # float64 weights + bias, per client

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        client_updates = []
        for cx, cy in zip(client_X, client_y):
            if len(np.unique(cy)) < 2:
                # client shard has only one class this round -> skip local SGD step,
                # just forward global weights unchanged
                client_updates.append((global_coef.copy(), global_intercept.copy(), len(cy)))
                continue
            coef, icpt, n = local_train(
                cx, cy, global_coef, global_intercept, classes, local_epochs=local_epochs
            )
            client_updates.append((coef, icpt, n))

        global_coef, global_intercept = federated_average(client_updates)
        round_time = time.time() - t0

        # Evaluate the aggregated global model (server-side, no client data needed)
        eval_model = SGDClassifier(loss="log_loss", random_state=RANDOM_SEED)
        eval_model.classes_ = classes
        eval_model.coef_ = global_coef
        eval_model.intercept_ = global_intercept
        preds = eval_model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        comm_bytes = bytes_per_update * num_clients * 2  # upload + download per round

        history["round"].append(rnd)
        history["accuracy"].append(acc)
        history["f1"].append(f1)
        history["comm_bytes"].append(comm_bytes)
        history["round_time_s"].append(round_time)

        if verbose:
            print(f"Round {rnd:2d}/{rounds} | acc={acc:.4f} | f1={f1:.4f} "
                  f"| comm={comm_bytes/1024:.1f} KB | time={round_time*1000:.1f} ms")

    final_preds = eval_model.predict(X_test)
    final_metrics = {
        "accuracy": accuracy_score(y_test, final_preds),
        "precision": precision_score(y_test, final_preds),
        "recall": recall_score(y_test, final_preds),
        "f1_score": f1_score(y_test, final_preds),
        "total_comm_bytes": sum(history["comm_bytes"]),
        "total_comm_KB": sum(history["comm_bytes"]) / 1024,
    }
    return history, final_metrics


# ---------------------------------------------------------------------------
# 5. Scalability test: vary number of clients (multi-cloud sites)
# ---------------------------------------------------------------------------
def scalability_test(client_counts=(2, 4, 8, 16), rounds=15):
    results = {}
    for n in client_counts:
        history, metrics = run_federated_learning(num_clients=n, rounds=rounds, verbose=False)
        results[n] = metrics
        print(f"Clients={n:2d} -> final acc={metrics['accuracy']:.4f}, "
              f"total comm={metrics['total_comm_KB']:.1f} KB")
    return results


# ---------------------------------------------------------------------------
# 6. Centralized baseline (for comparison -- what if we DIDN'T use FL?)
# ---------------------------------------------------------------------------
def centralized_baseline():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SGDClassifier(loss="log_loss", random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }


# ---------------------------------------------------------------------------
# Main: run everything and save plots + results
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("FEDERATED LEARNING FOR PRIVACY-PRESERVING MEDICAL DIAGNOSIS")
    print("=" * 70)

    print("\n[1] Centralized baseline (data pooled, for comparison only)")
    baseline = centralized_baseline()
    print(f"    Accuracy: {baseline['accuracy']:.4f} | F1: {baseline['f1_score']:.4f}")

    print("\n[2] Federated training across 4 simulated hospital/cloud clients")
    history, final_metrics = run_federated_learning(num_clients=4, rounds=15, non_iid=True)

    print("\n[3] Final Federated Model Metrics")
    for k, v in final_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n[4] Scalability test (varying number of cloud clients)")
    scale_results = scalability_test(client_counts=(2, 4, 8, 16), rounds=15)

    # ---- Save results as JSON for the report ----
    output = {
        "centralized_baseline": baseline,
        "federated_final_metrics": final_metrics,
        "federated_history": history,
        "scalability_results": {str(k): v for k, v in scale_results.items()},
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results.json")

    # ---- Plot 1: Accuracy over communication rounds ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["round"], history["accuracy"], marker="o", color="#2563eb", label="Federated model")
    axes[0].axhline(baseline["accuracy"], color="#dc2626", linestyle="--", label="Centralized baseline")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Model Accuracy vs. Federated Rounds")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cum_comm_kb = np.cumsum(history["comm_bytes"]) / 1024
    axes[1].plot(history["round"], cum_comm_kb, marker="s", color="#16a34a")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Cumulative Communication (KB)")
    axes[1].set_title("Communication Overhead Over Time")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")

    # ---- Plot 2: Scalability ----
    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    clients = list(scale_results.keys())
    accs = [scale_results[c]["accuracy"] for c in clients]
    comms = [scale_results[c]["total_comm_KB"] for c in clients]

    ax[0].bar([str(c) for c in clients], accs, color="#2563eb")
    ax[0].set_xlabel("Number of Cloud Clients")
    ax[0].set_ylabel("Final Test Accuracy")
    ax[0].set_title("Scalability: Accuracy vs. Number of Clients")
    ax[0].set_ylim(0.8, 1.0)
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].bar([str(c) for c in clients], comms, color="#16a34a")
    ax[1].set_xlabel("Number of Cloud Clients")
    ax[1].set_ylabel("Total Communication (KB)")
    ax[1].set_title("Scalability: Communication Cost vs. Number of Clients")
    ax[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("scalability_results.png", dpi=150)
    print("Saved scalability_results.png")

    print("\nDone.")
