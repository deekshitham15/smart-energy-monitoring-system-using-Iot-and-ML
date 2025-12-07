# evaluate_model.py
# Usage: put train.csv and model.json (optional) in same folder, then:
#   python evaluate_model.py
#
# Requires: pandas, numpy, scikit-learn
# Install: pip install pandas numpy scikit-learn

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# === CONFIG ===
TRAIN_CSV = Path("./train.csv")       # <-- change if your CSV is in a different path
MODEL_JSON = Path("./model.json")     # <-- your exported tree (optional)
OUT_PRED_CSV = Path("./predictions.csv")
N_SPLITS_CV = 5

# === helpers ===
def predict_with_tree_json(tree_node, features):
    """Evaluate a nested decision-tree JSON on a single feature dict."""
    node = tree_node
    while node is not None:
        if node.get("leaf"):
            return node.get("class")
        # feature name used in your JSON, e.g. "power" or "current"
        fname = node.get("feature")
        fval = float(features.get(fname, 0.0))
        threshold = float(node.get("threshold", 0.0))
        if fval <= threshold:
            node = node.get("left")
        else:
            node = node.get("right")
    return "Unknown"


# === main ===
if not TRAIN_CSV.exists():
    raise SystemExit(f"train.csv not found at {TRAIN_CSV}. Put your train.csv there or update TRAIN_CSV path.")

df = pd.read_csv(TRAIN_CSV)
expected_cols = {'power', 'voltage', 'current', 'label'}
if not expected_cols.issubset(df.columns):
    raise SystemExit(f"train.csv must contain columns: {expected_cols}. Found: {df.columns.tolist()}")

X = df[['power','voltage','current']].values
y = df['label'].values

if MODEL_JSON.exists():
    print("Found model.json — using it for predictions (cross-validated foldwise).")
    with open(MODEL_JSON, 'r') as f:
        tree_json = json.load(f)
    # cross-validated evaluation: predict each fold's test samples using the same tree
    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    y_pred = np.empty_like(y, dtype=object)
    for train_idx, test_idx in skf.split(X, y):
        for i in test_idx:
            feat = {'power': float(X[i,0]), 'voltage': float(X[i,1]), 'current': float(X[i,2])}
            y_pred[i] = predict_with_tree_json(tree_json, feat)
    model_used = "model.json (provided)"
else:
    print("model.json not found — training sklearn DecisionTreeClassifier for evaluation.")
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=N_SPLITS_CV)
    clf.fit(X, y)
    model_used = "sklearn DecisionTree (trained now)"

# Safety: fill None with "Unknown"
y_pred = [("Unknown" if v is None else v) for v in y_pred]

# Metrics
acc = accuracy_score(y, y_pred)
prec_macro = precision_score(y, y_pred, average='macro', zero_division=0)
rec_macro = recall_score(y, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
report = classification_report(y, y_pred, zero_division=0)
labels = np.unique(y)
cm = confusion_matrix(y, y_pred, labels=labels)


# Save results
out_df = df.copy()
out_df['predicted'] = y_pred
out_df.to_csv(OUT_PRED_CSV, index=False)

# Print
print("\n=== Evaluation summary ===")
print("Model used:", model_used)
print("Samples:", len(y))
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")
print(f"Recall (macro): {rec_macro:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print("\nClassification report:\n")
print(report)
print("Labels (true order):", labels.tolist())
print("\nConfusion matrix (rows=true labels, cols=predicted labels):\n")
print(cm)
print(f"\nPredictions saved to: {OUT_PRED_CSV}")

# Optional: show a few mismatches to inspect
mismatches = out_df[out_df['label'] != out_df['predicted']]
print(f"\nTotal mismatches: {len(mismatches)} (first 10 shown):")
print(mismatches.head(10).to_string(index=False))
