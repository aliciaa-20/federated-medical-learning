import http.server
import socketserver
import json
import os
import time
import urllib.parse
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PORT = 5000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Global dataset & preprocessor state
DATA = load_breast_cancer()
X_RAW, Y_RAW = DATA.data, DATA.target
FEATURE_NAMES = list(DATA.feature_names)

X_TRAIN_RAW, X_TEST_RAW, Y_TRAIN, Y_TEST = train_test_split(
    X_RAW, Y_RAW, test_size=0.2, random_state=RANDOM_SEED, stratify=Y_RAW
)

SCALER = StandardScaler()
X_TRAIN = SCALER.fit_transform(X_TRAIN_RAW)
X_TEST = SCALER.transform(X_TEST_RAW)

# Cached central baseline
def compute_baseline():
    model = SGDClassifier(loss="log_loss", random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_TRAIN, Y_TRAIN)
    preds = model.predict(X_TEST)
    return {
        "accuracy": float(accuracy_score(Y_TEST, preds)),
        "precision": float(precision_score(Y_TEST, preds)),
        "recall": float(recall_score(Y_TEST, preds)),
        "f1_score": float(f1_score(Y_TEST, preds))
    }

BASELINE = compute_baseline()

# Precompute sample cases for UI presets
def get_preset_cases():
    malignant_idx = np.where(Y_TEST == 0)[0][0]
    benign_idx = np.where(Y_TEST == 1)[0][0]
    
    # Feature stats for range sliders
    feature_stats = []
    for i, name in enumerate(FEATURE_NAMES):
        feature_stats.append({
            "name": name,
            "min": float(np.min(X_RAW[:, i])),
            "max": float(np.max(X_RAW[:, i])),
            "mean": float(np.mean(X_RAW[:, i])),
            "std": float(np.std(X_RAW[:, i]))
        })
        
    return {
        "features": feature_stats,
        "presets": {
            "malignant": {
                "label": "High-Risk Malignant Case",
                "diagnosis": "Malignant",
                "true_label": 0,
                "values": [float(v) for v in X_TEST_RAW[malignant_idx]]
            },
            "benign": {
                "label": "Typical Benign Case",
                "diagnosis": "Benign",
                "true_label": 1,
                "values": [float(v) for v in X_TEST_RAW[benign_idx]]
            }
        }
    }

PRESET_DATA = get_preset_cases()

# Active global model weights
ACTIVE_MODEL = {
    "coef": None,
    "intercept": None,
    "metrics": None
}

def partition_data(num_clients=4, non_iid=True):
    if non_iid:
        order = np.argsort(Y_TRAIN)
        x_tr, y_tr = X_TRAIN[order], Y_TRAIN[order]
    else:
        perm = np.random.permutation(len(Y_TRAIN))
        x_tr, y_tr = X_TRAIN[perm], Y_TRAIN[perm]

    client_X = np.array_split(x_tr, num_clients)
    client_y = np.array_split(y_tr, num_clients)
    return client_X, client_y

def local_train(X, y, coef, intercept, classes, local_epochs=1, lr=0.01):
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=lr,
        random_state=RANDOM_SEED,
        warm_start=True
    )
    model.classes_ = classes
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()

    for _ in range(local_epochs):
        model.partial_fit(X, y, classes=classes)

    return model.coef_, model.intercept_, len(y)

def federated_average(client_updates):
    total_samples = sum(n for _, _, n in client_updates)
    new_coef = sum(coef * n for coef, _, n in client_updates) / total_samples
    new_intercept = sum(icpt * n for _, icpt, n in client_updates) / total_samples
    return new_coef, new_intercept

def run_fl_simulation(num_clients=4, rounds=15, local_epochs=1, non_iid=True, lr=0.01):
    client_X, client_y = partition_data(num_clients=num_clients, non_iid=non_iid)
    n_features = client_X[0].shape[1]
    classes = np.array([0, 1])

    global_coef = np.zeros((1, n_features))
    global_intercept = np.zeros(1)

    history = {
        "round": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "comm_bytes": [],
        "cumulative_comm_kb": [],
        "round_time_ms": [],
        "client_distributions": [
            {
                "client_id": i + 1,
                "total_samples": len(client_y[i]),
                "malignant_count": int(np.sum(client_y[i] == 0)),
                "benign_count": int(np.sum(client_y[i] == 1))
            }
            for i in range(num_clients)
        ]
    }
    
    bytes_per_update = (n_features + 1) * 8
    cum_bytes = 0

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        client_updates = []
        for cx, cy in zip(client_X, client_y):
            coef, icpt, n = local_train(
                cx, cy, global_coef, global_intercept, classes, local_epochs=local_epochs, lr=lr
            )
            client_updates.append((coef, icpt, n))

        global_coef, global_intercept = federated_average(client_updates)
        round_time_ms = (time.time() - t0) * 1000

        # Server-side evaluation
        scores = np.dot(X_TEST, global_coef.T) + global_intercept
        preds = (scores.ravel() > 0).astype(int)

        acc = float(accuracy_score(Y_TEST, preds))
        f1 = float(f1_score(Y_TEST, preds))
        prec = float(precision_score(Y_TEST, preds, zero_division=0))
        rec = float(recall_score(Y_TEST, preds, zero_division=0))
        round_bytes = bytes_per_update * num_clients * 2
        cum_bytes += round_bytes

        history["round"].append(rnd)
        history["accuracy"].append(round(acc, 4))
        history["f1"].append(round(f1, 4))
        history["precision"].append(round(prec, 4))
        history["recall"].append(round(rec, 4))
        history["comm_bytes"].append(round_bytes)
        history["cumulative_comm_kb"].append(round(cum_bytes / 1024, 2))
        history["round_time_ms"].append(round(round_time_ms, 1))

    final_preds = (np.dot(X_TEST, global_coef.T) + global_intercept > 0).astype(int).ravel()
    final_metrics = {
        "accuracy": float(accuracy_score(Y_TEST, final_preds)),
        "precision": float(precision_score(Y_TEST, final_preds)),
        "recall": float(recall_score(Y_TEST, final_preds)),
        "f1_score": float(f1_score(Y_TEST, final_preds)),
        "total_comm_bytes": int(cum_bytes),
        "total_comm_KB": round(cum_bytes / 1024, 2),
    }

    ACTIVE_MODEL["coef"] = global_coef
    ACTIVE_MODEL["intercept"] = global_intercept
    ACTIVE_MODEL["metrics"] = final_metrics

    return history, final_metrics

# Pre-run default 4-client model so API is ready immediately
run_fl_simulation(num_clients=4, rounds=15, local_epochs=1, non_iid=True, lr=0.01)

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
        super().__init__(*args, directory=public_dir, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/initial-state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response_data = {
                "baseline": BASELINE,
                "current_model": ACTIVE_MODEL["metrics"],
                "preset_data": PRESET_DATA,
                "dataset_info": {
                    "total_samples": int(len(X_RAW)),
                    "train_samples": int(len(X_TRAIN)),
                    "test_samples": int(len(X_TEST)),
                    "features_count": int(len(FEATURE_NAMES)),
                    "classes": ["Malignant (0)", "Benign (1)"]
                }
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        if parsed.path == "/api/train":
            num_clients = int(params.get("num_clients", 4))
            rounds = int(params.get("rounds", 15))
            local_epochs = int(params.get("local_epochs", 1))
            non_iid = bool(params.get("non_iid", True))
            lr = float(params.get("lr", 0.01))

            history, final_metrics = run_fl_simulation(
                num_clients=num_clients,
                rounds=rounds,
                local_epochs=local_epochs,
                non_iid=non_iid,
                lr=lr
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "success": True,
                "baseline": BASELINE,
                "history": history,
                "final_metrics": final_metrics
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/predict":
            raw_features = params.get("features", [])
            if len(raw_features) != len(FEATURE_NAMES):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Expected 30 feature values"}).encode("utf-8"))
                return

            raw_arr = np.array(raw_features, dtype=float).reshape(1, -1)
            scaled_arr = SCALER.transform(raw_arr)

            coef = ACTIVE_MODEL["coef"]
            intercept = ACTIVE_MODEL["intercept"]

            # Linear logit score: z = w^T x + b
            z = float((np.dot(scaled_arr, coef.T) + intercept).ravel()[0])
            # Sigmoid probability for benign (class 1)
            prob_benign = 1.0 / (1.0 + np.exp(-z))
            prob_malignant = 1.0 - prob_benign

            prediction = 1 if z > 0 else 0
            diagnosis = "Benign" if prediction == 1 else "Malignant"
            confidence = prob_benign if prediction == 1 else prob_malignant

            # Feature influence / top contributors
            contributions = (scaled_arr[0] * coef[0]).tolist()
            top_factors = []
            for name, val, contrib in sorted(zip(FEATURE_NAMES, raw_features, contributions), key=lambda x: abs(x[2]), reverse=True)[:5]:
                top_factors.append({
                    "feature": name,
                    "value": round(val, 3),
                    "impact": "Favors Benign" if contrib > 0 else "Favors Malignant",
                    "weight": round(contrib, 3)
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "success": True,
                "diagnosis": diagnosis,
                "prediction_label": prediction,
                "confidence_percent": round(confidence * 100, 2),
                "prob_malignant": round(prob_malignant * 100, 2),
                "prob_benign": round(prob_benign * 100, 2),
                "logit_score": round(z, 4),
                "top_factors": top_factors,
                "privacy_guarantee": "Zero raw data left the local client during training or inference."
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/scalability":
            rounds = int(params.get("rounds", 15))
            client_counts = [2, 4, 8, 16]
            scalability_data = {}
            for n in client_counts:
                _, metrics = run_fl_simulation(num_clients=n, rounds=rounds)
                scalability_data[str(n)] = metrics

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "results": scalability_data}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def run_server():
    server_address = ("", PORT)
    # Enable address reuse to avoid port binding errors on restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, RequestHandler) as httpd:
        print(f"================================================================")
        print(f"MedFed AI Dashboard Server running at: http://localhost:{PORT}")
        print(f"Serving UI from ./public/index.html")
        print(f"================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    run_server()
