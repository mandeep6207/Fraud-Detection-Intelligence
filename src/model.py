"""Model builders for fraud detection."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Create a Logistic Regression classifier with sensible defaults."""
    return LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create a Random Forest classifier with sensible defaults."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def build_model_pipelines(preprocessor, random_state: int = 42) -> dict[str, Pipeline]:
    """Create full pipelines for each model with preprocessing included."""
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", build_logistic_regression(random_state=random_state)),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", build_random_forest(random_state=random_state)),
        ]
    )

    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": random_forest_pipeline,
    }
