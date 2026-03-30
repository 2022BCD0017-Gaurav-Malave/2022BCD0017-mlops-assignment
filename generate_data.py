import os
from sklearn.datasets import load_iris


def generate():
    os.makedirs("data", exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame
    feature_cols = [c for c in df.columns if c != "target"]

    v1_cols = feature_cols[:3] + ["target"]
    df.iloc[:80][v1_cols].to_csv("data/iris_v1.csv", index=False)
    df[feature_cols + ["target"]].to_csv("data/iris_v2.csv", index=False)
    print("Generated data/iris_v1.csv and data/iris_v2.csv")


if __name__ == "__main__":
    generate()
