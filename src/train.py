"""
Module to train best model
"""

# Dependencies

## Data Manipulation
import pandas as pd

## Machine Learning
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

## System & Files
import os
import json

## Utils
from tools.parse import init_parser
from tools.preprocessing import handle_missing_vals, cap_outliers
import joblib

def main(run_config_path: str, model_config_folder_path: str, model_name: str):
    # Parse arguments
    with open(run_config_path) as f:
        args = json.load(f)["train"]

    # Preprocess data
    df = pd.read_csv(args["train_data_path"])
    
    ## Drop irrelevant features
    df = df.drop(columns=args["drop_col"])
    
    ## Missing values
    df = handle_missing_vals(df)

    ## Duplicates
    l1 = len(df)
    df = df.drop_duplicates()
    l2 = len(df)
    print("Removing {} duplicates".format(l1 - l2))

    ## Quasi duplicates
    target = args["target"]
    mask_duplicated = df.duplicated(subset=df.columns.difference([target]), keep=False)
    df = df[~mask_duplicated]
    l1 = len(df)
    print("Removing {} quasi-duplicates".format(l2 - l1))

    mask_duplicated = df.duplicated(subset=df.columns.difference([target]), keep=False)
    assert not mask_duplicated.any().any(), "Duplicates remaining"

    ## Cap outliers
    df = cap_outliers(df,target)

    # Train model using json best params 
    model_configs_path = os.path.join(model_config_folder_path,model_name+".json")
    with open(model_configs_path) as f:
        model_configs = json.load(f)

    steps = []

    if model_configs["scaler_type"] == "robust":
        scaler = RobustScaler()
        steps.append(('scaler',scaler))
    
    if model_configs["pca_level"] is not None:
        pca = PCA(n_components=model_configs["pca_level"],random_state=model_configs["random_state"])
        steps.append(('pca',pca))
    
    if model_name == "lr":
        model = LogisticRegression(
            penalty=model_configs["penalty"],
            random_state=model_configs["random_state"],
            solver=model_configs["solver"],
            class_weight=model_configs["class_weight"],
            max_iter=model_configs["max_iter"],
            C=model_configs["C"],
        )
        steps.append(('classifier',model))

    pipeline = Pipeline(steps)

    X = df.drop(columns=target).values
    y = df[target]
    print("Training set shape : ", X.shape)

    pipeline.fit(X,y)
    print("Model trained")

    # Save model weights
    weights_folder_path = args["weights_folder_path"]
    os.makedirs(weights_folder_path, exist_ok=True)
    model_path = os.path.join(weights_folder_path, model_name + ".joblib")
    joblib.dump(pipeline, model_path)  # Joblib good format to save sklearn pipeline
    print(f"Model pipeline saved to {model_path}")


if __name__=="__main__":
    run_config_path = os.path.join(os.getcwd(),"src","run_configs.json")
    model_config_folder_path = os.path.join(os.getcwd(),"src","models","params")
    model_name = "lr"
    main(run_config_path,model_config_folder_path,model_name)


