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
from models.outlier_capper import Capper

## System & Files
import os
import json

## Utils
from tools.preprocessing import handle_duplicates, handle_missing_vals
import joblib

def main(run_config_path: str, model_config_folder_path: str, model_name: str):
    # Parse arguments
    with open(run_config_path) as f:
        args = json.load(f)["train"]

    df = pd.read_csv(args["train_data_path"])

    target = args ["target"]

    # Remove irrelevant column
    df = df.drop(columns=args["drop_col"])

    # Handling missing values
    df = handle_missing_vals(df)

    # Remove normal / quasi duplicates
    df = handle_duplicates(df, target)

    # Train model using json best params 
    model_configs_path = os.path.join(model_config_folder_path,model_name+".json")
    with open(model_configs_path) as f:
        model_configs = json.load(f)

    steps = []
    if model_configs["cap_factor"] is not None:
        capper = Capper(model_configs["cap_factor"])
        steps.append(('capper',capper))

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

    # Save test configs
    run_configs_test = args
    run_configs_test["model_name"] = model_name
    
    run_configs = {
        "train": args,
        "test": run_configs_test,
    }

    with open(run_config_path, "w") as f:
        json.dump(run_configs, f, indent=4)


if __name__=="__main__":
    run_config_path = os.path.join(os.getcwd(),"src","run_configs.json")
    model_config_folder_path = os.path.join(os.getcwd(),"src","models","params")
    model_name = "lr"
    main(run_config_path,model_config_folder_path,model_name)