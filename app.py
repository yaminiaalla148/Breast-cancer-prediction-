"""Local Flask application for the breast cancer prediction project."""
import math

from flask import Flask, jsonify, render_template, request, send_from_directory

from breast_cancer_model import (
    BASE_DIR,
    CLASS_NAMES,
    OUTPUT_DIR,
    train_and_evaluate,
)


FEATURE_GROUPS = [
    ("Mean measurements", "mean"),
    ("Standard error", "error"),
    ("Worst measurements", "worst"),
]
OUTPUT_IMAGES = {
    "class_distribution.png",
    "correlation_heatmap.png",
    "confusion_matrix.png",
    "pca_plot.png",
}


def create_app(analysis=None):
    """Create the dashboard app; inject trained analysis to simplify testing."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    analysis = analysis if analysis is not None else train_and_evaluate()
    feature_names = analysis["feature_names"]
    grouped_features = [
        {
            "title": title,
            "fields": [
                name
                for name in feature_names
                if (
                    name.endswith(" error")
                    if prefix == "error"
                    else name.startswith(prefix)
                )
            ],
        }
        for title, prefix in FEATURE_GROUPS
    ]

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            feature_groups=grouped_features,
            results=analysis["results"].to_dict(orient="records"),
            accuracy=analysis["accuracy"],
            form_values={},
            error=None,
            prediction=None,
        )

    @app.post("/predict")
    def predict():
        values = {}
        errors = []
        for name in feature_names:
            raw_value = request.form.get(name, "").strip()
            try:
                value = float(raw_value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError
                values[name] = value
            except ValueError:
                errors.append(name)

        if errors:
            message = (
                "Enter a valid, non-negative number for every measurement. "
                "Check: " + ", ".join(errors)
            )
            return render_template(
                "index.html",
                feature_groups=grouped_features,
                results=analysis["results"].to_dict(orient="records"),
                accuracy=analysis["accuracy"],
                form_values=request.form,
                error=message,
                prediction=None,
            ), 400

        prediction_value = int(
            analysis["model"].predict(
                [[values[name] for name in feature_names]]
            )[0]
        )
        is_malignant = prediction_value == 0
        prediction = {
            "label": CLASS_NAMES[prediction_value],
            "title": (
                "Cancer detected"
                if is_malignant
                else "No cancer detected"
            ),
            "detail": (
                "The model classified these measurements as a malignant "
                "(cancer-associated) pattern."
                if is_malignant
                else "The model classified these measurements as a benign "
                "(non-cancerous) pattern. This result cannot rule out cancer."
            ),
        }
        return render_template(
            "index.html",
            feature_groups=grouped_features,
            results=analysis["results"].to_dict(orient="records"),
            accuracy=analysis["accuracy"],
            form_values=values,
            error=None,
            prediction=prediction,
        )

    @app.get("/api/examples")
    def examples():
        return jsonify(analysis["examples"])

    @app.get("/outputs/<path:filename>")
    def output_image(filename):
        if filename not in OUTPUT_IMAGES:
            return "Not found", 404
        return send_from_directory(OUTPUT_DIR, filename)

    return app


if __name__ == "__main__":
    application = create_app()
    print("Dashboard ready at http://127.0.0.1:5000")
    application.run(host="127.0.0.1", port=5000, debug=False)
