"""
Module implementing logistic regression
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from models.outlier_capper import OutlierCapper

class LREstimator():
    def __init__(self, params: dict):
        self.name = "lr"
        if params["penalty"] is not None:
            self.name += "_" + params["penalty"]
        self.cv_folds = params["cv_folds"]
        self.beta = params["beta"]
        self.cap_factor = params["cap_factor"]
        self.scaler = params["scaler"]
        self.pca = params["pca"]
        if self.pca is not None:
            self.name += "_pca"
        model_params = {k:v for k,v in params.items() if k not in ["cv_folds","beta","scaler","pca","scaler_type","pca_level","model","n_trials","c_min","c_max","cap_factor"]}
        self.classifier = LogisticRegression(**model_params)

    def cross_validate(self, X_train: np.array, y_train: np.array):
        splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.classifier.random_state)
        score = 0.0
        self.cv_scores = {
            "train": {
                "precision": [],
                "recall": [],
                f"F{self.beta}": [],
            },
            "valid": {
                "precision": [],
                "recall": [],
                f"F{self.beta}": [],
            },
        }
        for i, (train_index,valid_index) in enumerate(splitter.split(X_train,y_train),start=1):
            X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
            X_valid_fold, y_valid_fold = X_train[valid_index], y_train[valid_index]

            self.outlier_cappers = []
            n_f = X_train.shape[1]
            for i in range(n_f):
                capper = OutlierCapper(self.cap_factor)
                capper.fit(X_train_fold[:,i])
                X_train_fold[:,i], capped_labels_train = capper.transform(X_train_fold[:,i])
                # X_train_fold = np.concatenate([X_train_fold, capped_labels_train.reshape(-1,1)],axis=1)

                X_valid_fold[:,i], capped_labels_valid = capper.transform(X_valid_fold[:,i])
                # X_valid_fold = np.concatenate([X_valid_fold, capped_labels_valid.reshape(-1,1)],axis=1)

                self.outlier_cappers.append(capper)

            X_train_fold = self.scaler.fit_transform(X_train_fold)
            X_valid_fold = self.scaler.transform(X_valid_fold)

            if self.pca is not None:
                # X_train_fold_capped = X_train_fold[:,n_f:]
                # X_valid_fold_capped = X_valid_fold[:,n_f:]
                # X_train_fold = np.concatenate([self.pca.fit_transform(X_train_fold[:,:n_f]),X_train_fold_capped],axis=1)
                # X_valid_fold = np.concatenate([self.pca.fit_transform(X_valid_fold[:,:n_f]),X_valid_fold_capped],axis=1)
                X_train_fold = self.pca.fit_transform(X_train_fold)
                X_valid_fold = self.pca.transform(X_valid_fold)
            
            self.classifier.fit(X_train_fold,y_train_fold)

            # Train scores
            train_preds = self.classifier.predict(X_train_fold)
            train_precision_fold = precision_score(y_train_fold,train_preds, zero_division=0)
            train_recall_fold = recall_score(y_train_fold,train_preds, zero_division=0)
            train_score_fold = fbeta_score(y_train_fold,train_preds,beta=self.beta, zero_division=0)

            self.cv_scores["train"]["precision"].append(train_precision_fold)
            self.cv_scores["train"]["recall"].append(train_recall_fold)
            self.cv_scores["train"][f"F{self.beta}"].append(train_score_fold)

            # Valid scores
            valid_preds = self.classifier.predict(X_valid_fold)
            valid_precision_fold = precision_score(y_valid_fold,valid_preds, zero_division=0)
            valid_recall_fold = recall_score(y_valid_fold,valid_preds, zero_division=0)
            valid_score_fold = fbeta_score(y_valid_fold,valid_preds,beta=self.beta, zero_division=0)

            self.cv_scores["valid"]["precision"].append(valid_precision_fold)
            self.cv_scores["valid"]["recall"].append(valid_recall_fold)
            self.cv_scores["valid"][f"F{self.beta}"].append(valid_score_fold)

            score += valid_score_fold
        return score / self.cv_folds
    
    def fit(self, X_train: np.array, y_train: np.array):
        self.outlier_cappers = []
        n_f = X_train.shape[1]
        for i in range(n_f):
            capper = OutlierCapper(self.cap_factor)
            capper.fit(X_train[:,i])
            X_train[:,i], capped_labels = capper.transform(X_train[:,i])
            # X_train = np.concatenate([X_train, capped_labels.reshape(-1,1)], axis=1)
            self.outlier_cappers.append(capper)

        X_train = self.scaler.fit_transform(X_train)

        if self.pca is not None:
            # X_train_capped = X_train[:,n_f:]
            # X_train = np.concatenate([self.pca.fit_transform(X_train[:,:n_f]),X_train_capped],axis=1)
            X_train = self.pca.fit_transform(X_train)

        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test: np.array, y_test: np.array):
        n_f = X_test.shape[1]
        for i in range(n_f):
            capper = self.outlier_cappers[i]
            X_test[:,i], capped_labels = capper.transform(X_test[:,i])
            # X_test = np.concatenate([X_test, capped_labels.reshape(-1,1)], axis=1)

        X_test = self.scaler.transform(X_test)

        if self.pca is not None:
            # X_test_capped = X_test[:,n_f:]
            # X_test = np.concatenate([self.pca.transform(X_test[:,:n_f]),X_test_capped],axis=1)
            X_test = self.pca.transform(X_test)

        test_preds = self.classifier.predict(X_test)
        cm = confusion_matrix(y_test, test_preds)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        fbeta = fbeta_score(y_test, test_preds, beta=self.beta, zero_division=0)
        return {
            "precision": precision,
            "recall": recall,
            f"F{self.beta}": fbeta,
            "confusion_matrix": cm,
        }