# train_and_export.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import json
import numpy as np

# Load labelled CSV
df = pd.read_csv('train.csv')  # columns: power, voltage, current, label
features = ['power', 'voltage', 'current']
X = df[features].values
y = df['label'].values

# Train small tree
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5)
clf.fit(X, y)

# Convert sklearn tree to nested dict JSON
def tree_to_dict(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            return {
                "feature": name,
                "threshold": float(threshold),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            # leaf -> class distribution
            values = tree_.value[node][0]
            # choose class with highest votes
            class_idx = int(values.argmax())
            classes = clf.classes_.tolist()
            return {"leaf": True, "class": classes[class_idx], "counts": values.tolist()}
    return recurse(0)

tree_json = tree_to_dict(clf, features)

with open('model.json', 'w') as f:
    json.dump(tree_json, f, indent=2)

print("Exported model.json")
