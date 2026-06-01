from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CLASS_NAMES, REPORTS_DIR
from real_domain import (
    REAL_DOMAIN_DATASET_PATH,
    REAL_DOMAIN_FEATURE_COLUMNS,
    REAL_DOMAIN_METADATA_PATH,
    REAL_DOMAIN_METRICS_PATH,
    REAL_DOMAIN_MODEL_NAME,
    REAL_DOMAIN_MODEL_PATH,
    build_real_domain_dataset,
    save_real_domain_metadata,
)

REAL_HOLDOUT_TEST_SIZE = 0.33
REAL_BENCHMARK_RANDOM_STATE = 9


def candidate_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale", probability=True)),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=42, min_samples_leaf=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=42, min_samples_leaf=1),
    }


def train_real_adapter() -> dict[str, object]:
    dataset = build_real_domain_dataset()
    X = dataset[REAL_DOMAIN_FEATURE_COLUMNS]
    y = dataset["label_index"]
    train_frame, test_frame = train_test_split(
        dataset,
        test_size=REAL_HOLDOUT_TEST_SIZE,
        random_state=REAL_BENCHMARK_RANDOM_STATE,
        stratify=dataset["label_index"],
    )

    X_train = train_frame[REAL_DOMAIN_FEATURE_COLUMNS]
    y_train = train_frame["label_index"]
    X_test = test_frame[REAL_DOMAIN_FEATURE_COLUMNS]
    y_test = test_frame["label_index"]

    best = None
    for model_name, model in candidate_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        if best is None or accuracy > best["test_accuracy"]:
            best = {
                "model_name": REAL_DOMAIN_MODEL_NAME,
                "estimator_name": model_name,
                "estimator": model,
                "test_accuracy": float(accuracy),
                "predictions": predictions,
            }

    report = classification_report(
        y_test,
        best["predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": REAL_DOMAIN_MODEL_NAME,
        "benchmark_type": "real_nasa_holdout",
        "estimator_name": best["estimator_name"],
        "train_samples": int(len(train_frame)),
        "test_samples": int(len(test_frame)),
        "test_accuracy": best["test_accuracy"],
        "random_state": REAL_BENCHMARK_RANDOM_STATE,
        "feature_columns": REAL_DOMAIN_FEATURE_COLUMNS,
    }

    joblib.dump(best["estimator"], REAL_DOMAIN_MODEL_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REAL_DOMAIN_METRICS_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=4, ensure_ascii=False)
    with open(REPORTS_DIR / f"classification_report_{REAL_DOMAIN_MODEL_NAME}.json", "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=4, ensure_ascii=False)
    save_real_domain_metadata(
        {
            "model_name": REAL_DOMAIN_MODEL_NAME,
            "feature_columns": REAL_DOMAIN_FEATURE_COLUMNS,
            "dataset_path": str(REAL_DOMAIN_DATASET_PATH),
            "metrics_path": str(REAL_DOMAIN_METRICS_PATH),
            "random_state": REAL_BENCHMARK_RANDOM_STATE,
            "estimator_name": best["estimator_name"],
            "test_accuracy": best["test_accuracy"],
        }
    )
    return metrics


def main() -> None:
    metrics = train_real_adapter()
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
