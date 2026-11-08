"""Reusable training and inference helpers for the breast cancer project."""
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CLASS_NAMES = {0: "malignant", 1: "benign"}


def load_dataset(path=DATASET_PATH):
    """Load the project dataset and validate its prediction target."""
    data = pd.read_csv(path)
    if "target" not in data.columns:
        raise ValueError("The dataset must contain a 'target' column.")
    if data.empty or data.isnull().any().any():
        raise ValueError("The dataset must contain rows with no missing values.")
    if set(data["target"].unique()) != set(CLASS_NAMES):
        raise ValueError("The target column must contain only 0 and 1.")
    return data


def train_and_evaluate(path=DATASET_PATH):
    """Train and evaluate the model set; return the tuned SVM for inference."""
    data = load_dataset(path)
    feature_names = data.drop(columns="target").columns.tolist()
    features = data[feature_names].to_numpy(dtype=float)
    target = data["target"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        cv_accuracy = cross_val_score(
            model, x_train, y_train, cv=cross_validation, scoring="accuracy"
        ).mean()
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "5-Fold CV Acc": cv_accuracy,
            }
        )

    svm_pipeline = make_pipeline(
        StandardScaler(), SVC(random_state=42)
    )
    search = GridSearchCV(
        svm_pipeline,
        {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", 0.01, 0.1],
        },
        cv=cross_validation,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    tuned_model = search.best_estimator_
    predictions = tuned_model.predict(x_test)
    results = pd.DataFrame(rows).round(4)
    examples = {
        CLASS_NAMES[int(label)]: profile[feature_names].to_dict()
        for label, profile in data.groupby("target").mean().iterrows()
    }

    return {
        "model": tuned_model,
        "feature_names": feature_names,
        "results": results,
        "accuracy": accuracy_score(y_test, predictions),
        "best_params": {
            key.removeprefix("svc__"): value
            for key, value in search.best_params_.items()
        },
        "classification_report": classification_report(
            y_test, predictions, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "data": data,
        "examples": examples,
    }


def create_analysis_outputs(analysis):
    """Write the model comparison and visualizations to the outputs folder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_DIR.mkdir(exist_ok=True)
    data = analysis["data"]
    results = analysis["results"]
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    sns.countplot(x="target", data=data)
    plt.title("Class distribution (0=malignant, 1=benign)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(data.iloc[:, :10].corr(), cmap="coolwarm")
    plt.title("Correlation of first 10 features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        analysis["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    plt.title("Confusion Matrix - Tuned SVM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    scaler = analysis["model"].named_steps["standardscaler"]
    scaled_train = scaler.transform(analysis["x_train"])
    scaled_all = scaler.transform(data.drop(columns="target").to_numpy(dtype=float))
    pca = PCA(n_components=2).fit(scaled_train)
    components = pca.transform(scaled_all)
    plt.figure(figsize=(6, 5))
    plt.scatter(
        components[:, 0],
        components[:, 1],
        c=data["target"],
        cmap="coolwarm",
        s=15,
    )
    variance = pca.explained_variance_ratio_.sum()
    plt.title(f"PCA (2 components, {variance:.0%} variance)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_plot.png")
    plt.close()
