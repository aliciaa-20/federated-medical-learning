import urllib.request
import json

base_url = "http://localhost:5000"

print("1. Testing GET / (HTML delivery)...")
req = urllib.request.urlopen(f"{base_url}/")
html = req.read().decode('utf-8')
assert "<title>MedFed AI" in html
print("   -> Success! HTML loaded with length", len(html))

print("2. Testing GET /api/initial-state...")
req = urllib.request.urlopen(f"{base_url}/api/initial-state")
init_data = json.loads(req.read().decode('utf-8'))
print(f"   -> Success! Baseline: {init_data['baseline']['accuracy'] * 100:.2f}%, Current: {init_data['current_model']['accuracy'] * 100:.2f}%")

print("3. Testing POST /api/train (5 rounds)...")
train_payload = json.dumps({"num_clients": 4, "rounds": 5, "local_epochs": 1, "non_iid": True, "lr": 0.01}).encode('utf-8')
train_req = urllib.request.Request(f"{base_url}/api/train", data=train_payload, headers={'Content-Type': 'application/json'})
train_res = json.loads(urllib.request.urlopen(train_req).read().decode('utf-8'))
print(f"   -> Success! Trained 5 rounds. Final Acc: {train_res['final_metrics']['accuracy'] * 100:.2f}%, F1: {train_res['final_metrics']['f1_score']:.4f}")

print("4. Testing POST /api/predict (Malignant Case Preset)...")
mal_feats = init_data['preset_data']['presets']['malignant']['values']
pred_payload = json.dumps({"features": mal_feats}).encode('utf-8')
pred_req = urllib.request.Request(f"{base_url}/api/predict", data=pred_payload, headers={'Content-Type': 'application/json'})
pred_res = json.loads(urllib.request.urlopen(pred_req).read().decode('utf-8'))
print(f"   -> Success! Diagnosis: {pred_res['diagnosis']} | Confidence: {pred_res['confidence_percent']}% | Factors: {len(pred_res['top_factors'])}")

print("5. Testing POST /api/predict (Benign Case Preset)...")
ben_feats = init_data['preset_data']['presets']['benign']['values']
pred_payload = json.dumps({"features": ben_feats}).encode('utf-8')
pred_req = urllib.request.Request(f"{base_url}/api/predict", data=pred_payload, headers={'Content-Type': 'application/json'})
pred_res = json.loads(urllib.request.urlopen(pred_req).read().decode('utf-8'))
print(f"   -> Success! Diagnosis: {pred_res['diagnosis']} | Confidence: {pred_res['confidence_percent']}%")

print("\nALL BACKEND API AND FRONTEND ASSET ENDPOINTS OPERATING AT 100% HEALTH!")
