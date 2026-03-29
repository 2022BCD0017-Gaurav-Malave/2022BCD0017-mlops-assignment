import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

STUDENT_NAME = "Gaurav Malave"      
ROLL_NO      = "2022BCD0017"   

dagshub.init(
    repo_owner="2022BCD0017-Gaurav-Malave",
    repo_name="2022BCD0017-mlops-assignment",
    mlflow=True
)

mlflow.set_experiment("2022BCD0017_experiment")

def load_data(csv_path, feature_set="all"):
    df = pd.read_csv(csv_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    if feature_set == "reduced":
        # Use only first 2 features (feature selection run)
        X = X.iloc[:, :2]
        selected = list(X.columns)
    else:
        selected = list(X.columns)

    return X, y, selected


def train_model(run_name, data_version, csv_path, model_type,
                n_estimators=100, max_depth=None, C=1.0,
                feature_set="all", experiment_name=None):

    if experiment_name is None:
        experiment_name = f"{ROLL_NO}_experiment"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # ── Load data ──
        X, y, selected_features = load_data(csv_path, feature_set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ── Build model ──
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:  # LogisticRegression
            model = LogisticRegression(C=C, max_iter=200, random_state=42)

        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted")

        # ── Log to MLflow ──
        mlflow.log_param("student_name",  STUDENT_NAME)
        mlflow.log_param("roll_no",       ROLL_NO)
        mlflow.log_param("data_version",  data_version)
        mlflow.log_param("model_type",    model_type)
        mlflow.log_param("feature_set",   feature_set)
        mlflow.log_param("selected_features", str(selected_features))
        mlflow.log_param("n_features",    len(selected_features))
        mlflow.log_param("dataset_size",  len(X))

        if model_type == "RandomForest":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth",    max_depth)
        else:
            mlflow.log_param("C", C)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        # ── Save metrics JSON ──
        metrics = {
            "name":         STUDENT_NAME,
            "roll_no":      ROLL_NO,
            "run_name":     run_name,
            "data_version": data_version,
            "model_type":   model_type,
            "accuracy":     round(acc, 4),
            "f1_score":     round(f1, 4),
            "features":     selected_features
        }
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/{run_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{run_name}] Accuracy={acc:.4f} | F1={f1:.4f}")
        print(f"  Features used: {selected_features}")

        # Save best model artifact
        joblib.dump(model,  "models/model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        return acc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",     type=int, default=0,
                        help="0=all runs, 1-5=specific run")
    parser.add_argument("--exp",     type=str,
                        default=f"{ROLL_NO}_experiment")
    args = parser.parse_args()

    os.makedirs("models",  exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    runs_config = [
        # Run 1: V1, RandomForest, all features, base config
        dict(run_name="Run1_RF_v1_base",
             data_version="v1", csv_path="data/iris_v1.csv",
             model_type="RandomForest",
             n_estimators=100, max_depth=None, feature_set="all"),

        # Run 2: V1, RandomForest, all features, different hyperparams
        dict(run_name="Run2_RF_v1_tuned",
             data_version="v1", csv_path="data/iris_v1.csv",
             model_type="RandomForest",
             n_estimators=50, max_depth=3, feature_set="all"),

        # Run 3: V2, RandomForest, all features, base config
        dict(run_name="Run3_RF_v2_base",
             data_version="v2", csv_path="data/iris_v2.csv",
             model_type="RandomForest",
             n_estimators=100, max_depth=None, feature_set="all"),

        # Run 4: V2, RandomForest, feature selection (reduced)
        dict(run_name="Run4_RF_v2_feat_select",
             data_version="v2", csv_path="data/iris_v2.csv",
             model_type="RandomForest",
             n_estimators=100, max_depth=None, feature_set="reduced"),

        # Run 5: V2, LogisticRegression, feature selection
        dict(run_name="Run5_LR_v2_feat_select",
             data_version="v2", csv_path="data/iris_v2.csv",
             model_type="LogisticRegression",
             C=0.5, feature_set="reduced"),
    ]

    if args.run == 0:
        for cfg in runs_config:
            train_model(**cfg, experiment_name=args.exp)
    else:
        train_model(**runs_config[args.run - 1], experiment_name=args.exp)