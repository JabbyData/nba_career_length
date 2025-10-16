"""
Module implementing complete pipeline to test model.
"""

# Dependencies
## Data manipulation
import pandas as pd
from tools.preprocessing import handle_duplicates, handle_missing_vals

## Linear Algebra
import numpy as np

## Machine Learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.pipeline import Pipeline

## System & Files
import os
import json
import joblib

## Visualization
from tabulate import tabulate

def score_classifier(dataset: np.array, classifier: Pipeline, labels: np.array, beta: float, n_splits: int=3) -> dict:

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision, recall and fbeta scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """
    kf = StratifiedKFold(n_splits=n_splits, random_state=classifier["classifier"].random_state, shuffle=True)
    confusion_mat = np.zeros((2,2))
    precision = 0.0
    recall = 0.0
    fbeta = 0.0
    for training_ids,test_ids in kf.split(dataset,labels):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        fbeta += fbeta_score(test_labels, predicted_labels, beta=beta)
    
    recall /= n_splits
    precision /= n_splits
    fbeta /= n_splits
    confusion_mat /= n_splits

    return {
        "recall": recall,
        "precision": precision,
        "fbeta": fbeta,
        "cm": confusion_mat,
    }


if __name__=="__main__":
    run_config_path = os.path.join(os.getcwd(),"src","run_configs.json")
    with open(run_config_path) as f:
        args = json.load(f)
    
    df = pd.read_csv(args["train_data_path"])

    target = args ["target"]
    if args["model_name"] == "lr":
        mode = "cap"
    
    pipeline_path = os.path.join(args["weights_folder_path"],args["model_name"]+".joblib")
    pipeline = joblib.load(pipeline_path)

    # Remove irrelevant column
    df = df.drop(columns=args["drop_col"])

    # Handling missing values
    df = handle_missing_vals(df)

    # Remove normal / quasi duplicates
    df = handle_duplicates(df, target)

    X = df.drop(columns=target).values
    y = df[target].values

    scores = score_classifier(X, pipeline, y, args["beta"])

    # Display metrics
    metrics_table = [
        ["Precision", scores["precision"]],
        ["Recall", scores["recall"]],
        [f"F{args['beta']}-score", scores["fbeta"]],
    ]
    print("Metrics (average):")
    print(tabulate(metrics_table, headers=["Metric", "Value"], floatfmt=".4f"))

    # Display Confusion matrix
    cm = scores["cm"]
    cm_table = [
        ["False", round(cm[0][0],2), round(cm[0][1],2)],
        ["True", round(cm[1][0],2), round(cm[1][1],2)]
    ]
    print("\nConfusion Matrix (average):")
    print(tabulate(cm_table, headers=["Real/Pred", "False", "True"]))

