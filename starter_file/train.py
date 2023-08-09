from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from azureml.core import Run, Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Add arguments to script
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for RandomForest')
parser.add_argument('--min_samples_split', type=int, default=10, help='Min samples split for RandomForest')
parser.add_argument('--min_samples_leaf', type=int, default=5, help='Min samples leaf for RandomForest')
args = parser.parse_args()

run = Run.get_context()
run.log('Number of estimators:', np.int(args.n_estimators))
run.log('Min samples split:', np.int(args.min_samples_split))
run.log('Min samples leaf:', np.int(args.min_samples_leaf))


workspace = run.experiment.workspace
dataset = Dataset.get_by_name(workspace, name='car evaluation data set') 

x = dataset.to_pandas_dataframe()

buying_dict = {'vhigh': 5, 'high': 4, 'med': 3, 'low': 2}
maint_dict = {'vhigh': 5, 'high': 4, 'med': 3, 'low': 2}
doors_dict = {'2': 2, '3': 3, '4': 4, '5more': 5}
persons_dict = {'2': 2, '4': 4, 'more': 5}
lug_boot_dict = {'small': 1, 'med': 3, 'big': 5}
safety_dict = {'low': 1, 'med': 3, 'high': 5}
class_dict = {'unacc':2, 'acc':3, 'good':4, 'vgood':5}
# カテゴリカル変数を数値に変換
x['buying'] = x['buying'].map(buying_dict)
x['maint'] = x['maint'].map(maint_dict)
x['doors'] = x['doors'].map(doors_dict)
x['persons'] = x['persons'].map(persons_dict)
x['lug_boot'] = x['lug_boot'].map(lug_boot_dict)
x['safety'] = x['safety'].map(safety_dict)
x['class'] = x['class'].map(class_dict)

y = x['class']
x = x.drop('class', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

model = RandomForestClassifier(n_estimators=args.n_estimators,
                               min_samples_split=args.min_samples_split,
                               min_samples_leaf=args.min_samples_leaf,
                               random_state=42)

model.fit(x_train, y_train)

# Calculate predicted probabilities instead of predicted labels
y_prob = model.predict_proba(x_test)

# Print shapes for debugging
print("Shape of y_test:", y_test.shape)
print("Shape of y_prob:", y_prob.shape)
print("Unique values in y_test:", np.unique(y_test))
print("Unique values in y_prob:", np.unique(y_prob))

accuracy = accuracy_score(y_test, y_prob)
run.log("accuracy", np.float(accuracy))

# Calculate ROC AUC score
auc_weighted = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
run.log('auc_weighted', np.float(auc_weighted))


# Save the model
os.makedirs('outputs', exist_ok=True)
joblib.dump(model, 'outputs/hyperdrive_model.pkl')

run.complete()