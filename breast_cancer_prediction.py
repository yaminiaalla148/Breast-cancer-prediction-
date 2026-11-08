"""Train, evaluate, and visualize the breast cancer prediction models.

Run ``python breast_cancer_prediction.py`` to create the analysis outputs.
Run ``python app.py`` to start the local prediction dashboard.
"""
from breast_cancer_model import create_analysis_outputs, train_and_evaluate


def main():
    analysis = train_and_evaluate()
    create_analysis_outputs(analysis)
    print("Dataset shape:", analysis["data"].shape)
    print("Missing values:", analysis["data"].isnull().sum().sum())
    print(analysis["results"].to_string(index=False), "\n")
    print("Best SVM params:", analysis["best_params"])
    print("Tuned SVM test accuracy:", f"{analysis['accuracy']:.4f}")
    print(analysis["classification_report"])
    print("\nDone. Plots and results saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()
