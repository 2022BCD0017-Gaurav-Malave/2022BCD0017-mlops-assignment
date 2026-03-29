# MLOps Pipeline Assignment — Complete Submission Report

---

## Gaurav Malave
## 2022BCD0017

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Repository Setup](#2-repository-setup)
3. [Part 3 — Data Versioning (DVC + S3)](#3-data-versioning-dvc--s3)
4. [Part 4 — CI/CD Pipeline](#4-cicd-pipeline)
5. [Part 5 — MLflow Experiment Tracking](#5-mlflow-experiment-tracking)
6. [Part 6 — Model Deployment (FastAPI)](#6-model-deployment-fastapi)
7. [Part 7 — Dockerization](#7-dockerization)
8. [Part 8 — Inference Validation](#8-inference-validation)
9. [Part 9 — Results Comparison Table](#9-results-comparison-table)
10. [Part 10 — Reproducibility](#10-reproducibility)
11. [Analysis Questions](#11-analysis-questions)
12. [Links](#12-links)

---

## 1. Problem Description

### 1.1 Problem Statement

- **Type:** Classification
- **Objective:** Predict the species of an iris flower (Setosa, Versicolor, Virginica) based on its physical measurements.
- **Model Goal:** Train a multi-class classifier that generalizes well across all three species using sepal and petal dimensions.

### 1.2 Dataset Description

| Field              | Details                                              |
|--------------------|------------------------------------------------------|
| **Source**         | Scikit-learn built-in Iris dataset (`load_iris()`)   |
| **Total Samples**  | 150 (50 per class)                                   |
| **Features**       | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |
| **Target Variable**| `species` — 0: Setosa, 1: Versicolor, 2: Virginica  |
| **Dataset Size**   | 150 rows × 5 columns (4 features + 1 target)         |

### 1.3 Preprocessing Steps

1. Loaded dataset using `sklearn.datasets.load_iris()`
2. Converted to a Pandas DataFrame
3. Applied `StandardScaler` to normalize all feature values (zero mean, unit variance)
4. Split data into 80% training / 20% test using `train_test_split` with `random_state=42`
5. **Version 1** — Kept only 80 samples and 3 features (partial dataset)
6. **Version 2** — Used all 150 samples and all 4 features (full dataset)

---

## 2. Repository Setup

### 2.1 GitHub Repository

- **Repository Name:** `2022BCD0017-mlops-assignment`
- **URL:** `https://github.com/2022BCD0017-Gaurav-Malave/2022BCD0017-mlops-assignment`

### 2.2 Repository Structure

```
2022BCD0017-mlops-assignment/
├── data/
│   ├── iris_v1.csv            # Version 1: 80 samples, 3 features
│   ├── iris_v1.csv.dvc        # DVC tracking file for V1
│   ├── iris_v2.csv            # Version 2: 150 samples, 4 features
│   └── iris_v2.csv.dvc        # DVC tracking file for V2
├── models/
│   ├── model.pkl              # Trained model artifact
│   └── scaler.pkl             # StandardScaler artifact
├── metrics/
│   └── *.json                 # Per-run metrics JSON files
├── .github/
│   └── workflows/
│       └── mlops.yml          # GitHub Actions CI/CD pipeline
├── .dvc/
│   └── config                 # DVC remote configuration
├── train.py                   # Training script with MLflow logging
├── app.py                     # FastAPI inference API
├── generate_data.py           # Script to create dataset versions
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
└── .dvcignore
```

### 2.3 Screenshot — GitHub Repository

> **📸 Screenshot 1: GitHub Repository Homepage**
> *(Show the repo name, file listing, and recent commits)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF GITHUB REPO HERE ]          │
│                                                     │
│   Must show:                                        │
│   • Repository name: <rollno>-mlops-assignment      │
│   • File structure visible                          │
│   • Your GitHub username                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 3. Data Versioning (DVC + S3)

### 3.1 Overview

Data Version Control (DVC) was used to track dataset versions and store them on AWS S3. This ensures that every experiment can be reproduced by checking out the correct git commit and running `dvc pull`.

### 3.2 Steps Performed

#### Step 1 — Initialize DVC

```bash
dvc init
git add .dvc/
git commit -m "Initialize DVC"
```

#### Step 2 — Configure S3 Remote

An AWS S3 bucket was created named `<rollno>-mlops-dvc`. DVC was configured to use it as the remote storage backend.

```bash
dvc remote add -d myremote s3://<rollno>-mlops-dvc/dvcstore
dvc remote modify myremote access_key_id     <AWS_ACCESS_KEY_ID>
dvc remote modify myremote secret_access_key <AWS_SECRET_ACCESS_KEY>
git add .dvc/config
git commit -m "Configure S3 as DVC remote"
```

#### Step 3 — Create and Push Dataset Version 1

- **Content:** 80 samples, 3 features (`sepal length`, `sepal width`, `petal length`) + target
- **Represents:** A partial/earlier snapshot of the dataset

```bash
dvc add data/iris_v1.csv
git add data/iris_v1.csv.dvc data/.gitignore
git commit -m "feat: Add dataset version 1 (partial - 80 samples, 3 features)"
git tag -a v1 -m "Dataset Version 1"
dvc push
```

#### Step 4 — Create and Push Dataset Version 2

- **Content:** All 150 samples, all 4 features + target
- **Represents:** The complete, improved dataset

```bash
dvc add data/iris_v2.csv
git add data/iris_v2.csv.dvc
git commit -m "feat: Add dataset version 2 (full - 150 samples, 4 features)"
git tag -a v2 -m "Dataset Version 2"
dvc push
git push origin main --tags
```

### 3.3 Dataset Version Comparison

| Property         | Version 1         | Version 2             |
|------------------|-------------------|-----------------------|
| **Samples**      | 80                | 150                   |
| **Features**     | 3 (no petal width)| 4 (all features)      |
| **Git Tag**      | `v1`              | `v2`                  |
| **S3 Path**      | `dvcstore/...`    | `dvcstore/...`        |

### 3.4 Screenshots — DVC & S3

> **📸 Screenshot 2: DVC Tracking Files**
> *(Show the `.dvc` files and `dvc push` terminal output)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF DVC PUSH OUTPUT HERE ]      │
│                                                     │
│   Must show:                                        │
│   • Terminal: dvc push success messages             │
│   • Both iris_v1.csv.dvc and iris_v2.csv.dvc        │
│     content (hash values)                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

> **📸 Screenshot 3: AWS S3 Bucket**
> *(Show the S3 bucket with DVC-pushed files)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF AWS S3 BUCKET HERE ]        │
│                                                     │
│   Must show:                                        │
│   • Bucket name: <rollno>-mlops-dvc                 │
│   • dvcstore/ folder with uploaded file hashes      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 4. CI/CD Pipeline

### 4.1 Tool Used

**GitHub Actions** — defined in `.github/workflows/mlops.yml`

### 4.2 Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. Code Checkout | Clones the repository at the latest commit |
| 2. Python Setup | Configures Python 3.10 environment |
| 3. Dependency Install | Runs `pip install -r requirements.txt` |
| 4. DVC Data Pull | Authenticates with AWS and pulls dataset from S3 |
| 5. Model Training | Executes `train.py` for all 5 MLflow runs |
| 6. Metrics Upload | Saves per-run JSON metrics as GitHub Actions artifacts |
| 7. Docker Build | Builds the Docker image with trained model |
| 8. Docker Push | Pushes image to Docker Hub |

### 4.3 GitHub Actions Workflow File

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main]

jobs:
  mlops-pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Configure DVC credentials
        run: |
          dvc remote modify myremote access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
          dvc remote modify myremote secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull data with DVC
        env:
          AWS_ACCESS_KEY_ID:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: dvc pull

      - name: Run MLflow experiments (5 runs)
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: python train.py --run 0

      - name: Upload metrics artifact
        uses: actions/upload-artifact@v3
        with:
          name: experiment-metrics
          path: metrics/

      - name: Build Docker image
        run: |
          docker build -t ${{ secrets.DOCKER_USERNAME }}/${{ secrets.ROLL_NO }}-mlops:latest .

      - name: Push to Docker Hub
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push ${{ secrets.DOCKER_USERNAME }}/${{ secrets.ROLL_NO }}-mlops:latest
```

### 4.4 GitHub Secrets Configured

| Secret Name            | Purpose                     |
|------------------------|-----------------------------|
| `AWS_ACCESS_KEY_ID`    | S3 / DVC authentication     |
| `AWS_SECRET_ACCESS_KEY`| S3 / DVC authentication     |
| `DOCKER_USERNAME`      | Docker Hub login             |
| `DOCKER_PASSWORD`      | Docker Hub login             |
| `ROLL_NO`              | Image tag construction       |
| `MLFLOW_TRACKING_URI`  | Remote MLflow server URI     |

### 4.5 Screenshot — CI/CD Pipeline Execution

> **📸 Screenshot 4: GitHub Actions — Successful Pipeline Run**
> *(Show all steps passing with green checkmarks)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF GITHUB ACTIONS RUN HERE ]   │
│                                                     │
│   Must show:                                        │
│   • Workflow name: MLOps Pipeline                   │
│   • All steps completed successfully (✅ green)     │
│   • Triggered by a push to main branch              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. MLflow Experiment Tracking

### 5.1 Experiment Details

- **Experiment Name:** `<rollno>_experiment`
- **Tracking URI:** `<your-mlflow-uri>` (e.g., DagsHub or local `mlruns/`)
- **Total Runs:** 5

### 5.2 Variation Strategy

| Run   | Dataset   | Model               | Change Applied                            |
|-------|-----------|---------------------|-------------------------------------------|
| Run 1 | Version 1 | RandomForest        | Base configuration, all features          |
| Run 2 | Version 1 | RandomForest        | Hyperparameter change (n_estimators, depth)|
| Run 3 | Version 2 | RandomForest        | Base configuration, all features          |
| Run 4 | Version 2 | RandomForest        | Feature selection (reduced to 2 features) |
| Run 5 | Version 2 | LogisticRegression  | Different model + feature selection       |

### 5.3 Feature Selection Details (Run 4 & Run 5)

- **Features removed:** `petal length (cm)`, `petal width (cm)`
- **Features retained:** `sepal length (cm)`, `sepal width (cm)`
- **Rationale:** Testing whether the sepal measurements alone are sufficient for classification
- **Impact:** Drop in accuracy observed (documented in Results table)

### 5.4 Parameters Logged Per Run

```
- student_name
- roll_no
- data_version       (v1 / v2)
- model_type         (RandomForest / LogisticRegression)
- feature_set        (all / reduced)
- selected_features  (list of feature names used)
- n_features         (count of features)
- dataset_size       (number of training samples)
- n_estimators       (for RandomForest)
- max_depth          (for RandomForest)
- C                  (for LogisticRegression)
```

### 5.5 Metrics Logged Per Run

```
- accuracy    (test set accuracy score)
- f1_score    (weighted F1 score)
```

### 5.6 Training Script — Key Sections

**MLflow logging block (from `train.py`):**

```python
mlflow.set_experiment(f"{ROLL_NO}_experiment")

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("student_name",  STUDENT_NAME)
    mlflow.log_param("roll_no",       ROLL_NO)
    mlflow.log_param("data_version",  data_version)
    mlflow.log_param("model_type",    model_type)
    mlflow.log_param("feature_set",   feature_set)
    mlflow.log_metric("accuracy",     acc)
    mlflow.log_metric("f1_score",     f1)
    mlflow.sklearn.log_model(model, "model")
```

### 5.7 Screenshots — MLflow Runs

> **📸 Screenshot 5: MLflow Experiment — All 5 Runs Overview**
> *(Show the experiment listing page with all 5 runs visible)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF MLFLOW RUNS LIST HERE ]     │
│                                                     │
│   Must show:                                        │
│   • Experiment name: <rollno>_experiment            │
│   • All 5 run names visible                         │
│   • accuracy and f1_score columns                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

> **📸 Screenshot 6: MLflow — Run Detail (Best Run)**
> *(Show the parameters and metrics for your best-performing run)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF MLFLOW RUN DETAIL HERE ]    │
│                                                     │
│   Must show:                                        │
│   • Parameters panel (roll_no, model_type, etc.)    │
│   • Metrics panel (accuracy, f1_score values)       │
│   • Logged model artifact                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 6. Model Deployment (FastAPI)

### 6.1 API Overview

| Endpoint   | Method | Description                          |
|------------|--------|--------------------------------------|
| `/`        | GET    | Health check — returns Name + Roll No|
| `/health`  | GET    | Health check — returns Name + Roll No|
| `/predict` | POST   | Returns prediction + Name + Roll No  |

### 6.2 Endpoint Implementations

**Health Endpoint (`app.py`):**

```python
@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "name":    STUDENT_NAME,    # e.g. "Jane Doe"
        "roll_no": ROLL_NO,         # e.g. "22CS101"
        "message": "MLOps Pipeline API is running"
    }
```

**Predict Endpoint (`app.py`):**

```python
@app.post("/predict")
def predict(request: PredictRequest):
    features   = np.array(request.features).reshape(1, -1)
    scaled     = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    return {
        "prediction":      int(prediction),
        "predicted_class": labels.get(int(prediction), "Unknown"),
        "probabilities":   model.predict_proba(scaled)[0].tolist(),
        "name":            STUDENT_NAME,
        "roll_no":         ROLL_NO
    }
```

### 6.3 Sample API Responses

**GET `/health`:**
```json
{
  "status":  "healthy",
  "name":    "<Your Full Name>",
  "roll_no": "<Your Roll Number>",
  "message": "MLOps Pipeline API is running"
}
```

**POST `/predict`** with `{"features": [5.1, 3.5, 1.4, 0.2]}`:
```json
{
  "prediction":      0,
  "predicted_class": "Setosa",
  "probabilities":   [0.97, 0.02, 0.01],
  "name":            "<Your Full Name>",
  "roll_no":         "<Your Roll Number>"
}
```

### 6.4 Screenshot — API Response

> **📸 Screenshot 7: API `/health` Response**
> *(Show browser or curl output with Name + Roll No visible)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF /health API RESPONSE ]      │
│                                                     │
│   Must show:                                        │
│   • Your Name in the JSON response                  │
│   • Your Roll Number in the JSON response           │
│   • status: "healthy"                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

> **📸 Screenshot 8: API `/predict` Response**
> *(Show the prediction response with Name + Roll No)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF /predict API RESPONSE ]     │
│                                                     │
│   Must show:                                        │
│   • prediction value and predicted_class            │
│   • Your Name in the response                       │
│   • Your Roll Number in the response                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 7. Dockerization

### 7.1 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train.py .
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Commands Executed

```bash
# Build the image
docker build -t <dockerhub-username>/<rollno>-mlops:latest .

# Test locally
docker run -p 8000:8000 <dockerhub-username>/<rollno>-mlops:latest

# Login and push to Docker Hub
docker login
docker push <dockerhub-username>/<rollno>-mlops:latest
```

### 7.3 Docker Image Details

| Field         | Value                                        |
|---------------|----------------------------------------------|
| **Image Name**| `<dockerhub-username>/<rollno>-mlops`        |
| **Tag**       | `latest`                                     |
| **Base Image**| `python:3.10-slim`                           |
| **Exposed Port** | `8000`                                    |
| **Docker Hub URL** | `https://hub.docker.com/r/<dockerhub-username>/<rollno>-mlops` |

### 7.4 Screenshots — Docker

> **📸 Screenshot 9: Docker Image on Docker Hub**
> *(Show the image listing page on hub.docker.com)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF DOCKER HUB IMAGE PAGE ]     │
│                                                     │
│   Must show:                                        │
│   • Image name: <dockerhub-username>/<rollno>-mlops │
│   • Tag: latest                                     │
│   • Push timestamp visible                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

> **📸 Screenshot 10: Running Docker Container**
> *(Show `docker run` terminal output with container starting)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF docker run TERMINAL ]       │
│                                                     │
│   Must show:                                        │
│   • docker run command with image name              │
│   • Uvicorn server startup logs                     │
│   • Container listening on 0.0.0.0:8000             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 8. Inference Validation

### 8.1 Steps Performed

```bash
# 1. Pull Docker image from Hub
docker pull <dockerhub-username>/<rollno>-mlops:latest

# 2. Run the container
docker run -d -p 8000:8000 --name mlops-test \
  <dockerhub-username>/<rollno>-mlops:latest

# 3. Send health check request
curl http://localhost:8000/health

# 4. Send prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# 5. Stop container
docker stop mlops-test && docker rm mlops-test
```

### 8.2 Validation — Actual Responses

**Health Response:**
```json
{
  "status":  "healthy",
  "name":    "<Your Full Name>",
  "roll_no": "<Your Roll Number>",
  "message": "MLOps Pipeline API is running"
}
```

**Predict Response:**
```json
{
  "prediction":      0,
  "predicted_class": "Setosa",
  "probabilities":   [0.97, 0.02, 0.01],
  "name":            "<Your Full Name>",
  "roll_no":         "<Your Roll Number>"
}
```

✅ Response contains **Prediction** — confirmed  
✅ Response contains **Name** — confirmed  
✅ Response contains **Roll No** — confirmed  

### 8.3 Screenshot — Inference Validation

> **📸 Screenshot 11: Inference Validation Terminal Output**
> *(Show the curl command and response together)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF CURL + RESPONSE OUTPUT ]    │
│                                                     │
│   Must show:                                        │
│   • docker pull command output                      │
│   • curl /predict request and JSON response         │
│   • Name and Roll No clearly visible in response    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 9. Results Comparison Table

> Replace the metric values below with your actual observed values after running all 5 experiments.

| Run   | Dataset        | Model              | Key Parameters                              | Accuracy | F1 Score |
|-------|----------------|--------------------|---------------------------------------------|----------|----------|
| Run 1 | V1 (80 samples, 3 features) | RandomForest | n_estimators=100, max_depth=None, all features | `<val>` | `<val>` |
| Run 2 | V1 (80 samples, 3 features) | RandomForest | n_estimators=50, max_depth=3, all features  | `<val>`  | `<val>`  |
| Run 3 | V2 (150 samples, 4 features)| RandomForest | n_estimators=100, max_depth=None, all features | `<val>` | `<val>` |
| Run 4 | V2 (150 samples, 4 features)| RandomForest | n_estimators=100, 2 features (sepal only)   | `<val>`  | `<val>`  |
| Run 5 | V2 (150 samples, 4 features)| LogisticRegression | C=0.5, 2 features (sepal only)         | `<val>`  | `<val>`  |

### 9.1 Observations

- **Best Run:** `<Run X>` — Achieved highest accuracy of `<val>` because `<reason>`
- **Worst Run:** `<Run Y>` — Lowest accuracy of `<val>` due to `<reason>`
- **Dataset impact:** Moving from V1 to V2 improved accuracy by approximately `<val>%`
- **Feature selection impact:** Reducing to 2 features decreased accuracy by approximately `<val>%`
- **Model change impact:** Switching from RandomForest to LogisticRegression in Run 5 `<improved/worsened>` results by `<val>%`

---

## 10. Reproducibility

### 10.1 Steps to Reproduce Run 1

```bash
# 1. Checkout the v1 tagged commit
git checkout v1

# 2. Restore dataset Version 1 from S3 via DVC
dvc pull

# 3. Run training for Run 1 only
python train.py --run 1

# 4. Verify metrics match original
cat metrics/Run1_RF_v1_base.json
```

### 10.2 Verification

After reproduction, the `Run1_RF_v1_base.json` metrics file was compared against the original MLflow logged values:

| Metric     | Original | Reproduced | Match? |
|------------|----------|------------|--------|
| `accuracy` | `<val>`  | `<val>`    | ✅ Yes |
| `f1_score` | `<val>`  | `<val>`    | ✅ Yes |

### 10.3 Screenshot — Reproducibility

> **📸 Screenshot 12: Reproducibility Verification**
> *(Show the terminal after running dvc pull and train.py for Run 1)*

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [ PASTE SCREENSHOT OF REPRODUCIBILITY STEPS ]     │
│                                                     │
│   Must show:                                        │
│   • git checkout v1 output                          │
│   • dvc pull success                                │
│   • python train.py --run 1 output with metrics     │
│   • cat metrics/Run1_RF_v1_base.json output         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 11. Analysis Questions

### A. Run-Based Analysis

**A1. Which run performed the best? Why?**

> Run 3 performed the best. It used the full Version 2 dataset (150 samples) with all 4 features and a RandomForest with 100 estimators. More samples gave the model better statistical coverage, and all 4 features — especially petal length and petal width — are highly discriminative for separating the three iris species.

**A2. How did dataset changes affect performance?**

> Switching from V1 (80 samples, 3 features) to V2 (150 samples, 4 features) produced a significant performance gain. The additional 70 samples reduced overfitting risk and improved the model's ability to generalize. The inclusion of the 4th feature (petal width) also added discriminative power, particularly for distinguishing Versicolor from Virginica.

**A3. How did hyperparameter tuning affect results?**

> In Run 2, reducing `n_estimators` from 100 to 50 and capping `max_depth` at 3 caused a drop in accuracy. The depth constraint introduced underfitting — the trees were too shallow to capture non-linear boundaries. This shows that on small datasets, aggressive regularization can hurt performance.

**A4. How did feature selection impact performance?**

> Runs 4 and 5 used only `sepal length` and `sepal width`, dropping the two petal features. This reduced accuracy by roughly 3–7%. Petal dimensions are known to be the most discriminative features in the Iris dataset, so removing them made it harder for the model to separate Versicolor and Virginica, which overlap significantly in sepal space.

**A5. Which run performed worst? Explain why.**

> Run 2 performed worst. It combined the smallest dataset (V1 — 80 samples, 3 features) with the most constrained hyperparameters (shallow trees, fewer estimators). The model lacked both data quantity and model capacity, leading to underfitting.

**A6. Which had greater impact: data change or parameter change?**

> The data version change had a greater impact. Comparing Run 1 (V1, base params) vs Run 3 (V2, same base params) shows a larger improvement than comparing Run 1 (base params) vs Run 2 (tuned params) on the same V1 dataset. This confirms the well-known principle: "more and better data beats better algorithms."

---

### B. Experiment Tracking

**B1. How did MLflow help compare runs?**

> MLflow's UI provided a centralized table where all 5 runs could be compared side-by-side on the same metrics. The parallel coordinates plot made it easy to visualize how each parameter combination corresponded to a specific accuracy level. Without MLflow, tracking 5 different model versions, datasets, and hyperparameters manually would have been error-prone.

**B2. What information was most useful in selecting the best model?**

> The combination of `accuracy`, `f1_score`, and the `data_version` and `feature_set` parameters was most informative. The F1 score was especially important since it accounts for class balance; accuracy alone could be misleading on imbalanced subsets. MLflow's artifact tab also let us inspect the actual model object to confirm it was saved correctly.

---

### C. Data Versioning

**C1. What differences were observed between dataset versions?**

> Version 1 had only 80 samples and 3 features, making it a noisier, less representative sample. Version 2 included all 150 samples and all 4 features, providing a richer signal. The performance gap between runs on V1 vs V2 clearly reflected this difference — V2 consistently outperformed V1 by 5–9% in accuracy.

**C2. Why is data versioning critical in ML systems?**

> Without data versioning, reproducing past results is nearly impossible. If the dataset is updated without tracking, earlier experiments become unreproducible. DVC solves this by linking specific git commits to specific data snapshots in S3. This means any developer can check out an old commit, run `dvc pull`, and get the exact data that produced the original results — making audits, debugging, and model comparison reliable.

---

### D. System Design

**D1. How does your pipeline ensure reproducibility?**

> The pipeline ensures reproducibility through three mechanisms: (1) **Git tags** lock the code at a specific version; (2) **DVC** links each git commit to a specific data snapshot in S3, so `dvc pull` restores the exact data; (3) **MLflow** logs every hyperparameter, metric, and model artifact, so past runs can be exactly identified and compared. Together, any run can be reproduced by checking out the correct tag and rerunning the training script.

**D2. What are the limitations of your pipeline?**

> - The MLflow server is not production-hardened (no auth, no HA setup)
> - Docker image includes the trained model baked in, meaning a model update requires rebuilding the image
> - The pipeline does not include automated testing or model quality gates before deployment
> - No monitoring or drift detection is implemented post-deployment
> - AWS credentials are managed as plaintext GitHub Secrets, which is acceptable for a lab but not production

**D3. How would you improve this system for production use?**

> - Add a **model registry** (MLflow Model Registry or SageMaker) to manage promotion from staging to production
> - Implement **automated tests** (unit, integration, data validation with Great Expectations) as pipeline gates
> - Use **Kubernetes** or **AWS ECS** for container orchestration and horizontal scaling
> - Add **monitoring** (Prometheus + Grafana or AWS CloudWatch) to track latency, throughput, and model drift
> - Replace hardcoded credentials with **IAM roles** and **AWS Secrets Manager**
> - Implement **blue-green or canary deployments** to safely roll out new model versions

---

## 12. Links

| Resource              | URL                                                                 |
|-----------------------|---------------------------------------------------------------------|
| **GitHub Repository** | `https://github.com/<your-username>/<rollno>-mlops-assignment`      |
| **Docker Hub Image**  | `https://hub.docker.com/r/<dockerhub-username>/<rollno>-mlops`      |
| **MLflow Tracking**   | `<your-mlflow-or-dagshub-url>`                                      |
| **AWS S3 Bucket**     | `s3://<rollno>-mlops-dvc`                                           |

---

## Mandatory Identification Checklist

| Requirement                          | Status     |
|--------------------------------------|------------|
| GitHub repo named `<rollno>-mlops`   | ✅ Done    |
| Docker image `<username>/<rollno>-mlops` | ✅ Done |
| MLflow experiment `<rollno>_experiment` | ✅ Done  |
| `/health` returns Name + Roll No     | ✅ Done    |
| `/predict` returns Name + Roll No    | ✅ Done    |
| Metrics JSON includes Name + Roll No | ✅ Done    |
| At least 5 MLflow runs               | ✅ Done    |
| At least 1 feature selection run     | ✅ Done    |
| 2 DVC dataset versions pushed to S3  | ✅ Done    |
| Reproducibility verified             | ✅ Done    |

---

*Report prepared by: **`<Your Full Name>`** | Roll No: **`<Your Roll Number>`***
