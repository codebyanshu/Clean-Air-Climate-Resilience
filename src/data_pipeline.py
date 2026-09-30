from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .project_paths import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORTS_DIR, load_config



def pollutant_columns() -> list[str]:
    config = load_config()
    columns = config.get("data", {}).get("pollutant_columns")
    if columns:
        return [str(col) for col in columns]
    return [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
        "Benzene", "Toluene", "Xylene"
    ]


def load_raw_dataset(filename: str = "city_day.csv") -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    if "City" in cleaned.columns:
        cleaned = cleaned.sort_values(["City", "Date"] if "Date" in cleaned.columns else ["City"]).copy()
    cols = pollutant_columns()
    for column in cols:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in cols:
        if column in cleaned.columns:
            cleaned[column] = cleaned.groupby("City", dropna=False)[column].transform(
                lambda values: values.interpolate(limit_direction="both")
            )
            cleaned[column] = cleaned[column].fillna(cleaned.groupby("City", dropna=False)[column].transform("median"))
    if "AQI" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["AQI"]).copy()
    cleaned = cleaned.drop_duplicates()
    if {"City", "Date"}.issubset(cleaned.columns):
        cleaned = cleaned.drop_duplicates(subset=["City", "Date"], keep="first")
    for column in [*cols, "AQI"]:
        if column in cleaned.columns and cleaned[column].notna().any():
            lower, upper = cleaned[column].quantile([0.01, 0.99])
            cleaned[column] = cleaned[column].clip(lower, upper)
    return cleaned


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    if "Date" in features.columns:
        features["month"] = features["Date"].dt.month
    return features


def save_clean_data(df: pd.DataFrame, output_path: Path | str | None = None) -> Path:
    target = Path(output_path) if output_path is not None else INTERIM_DIR / "city_day_clean.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def save_processed_data(df: pd.DataFrame, output_path: Path | str | None = None) -> Path:
    target = Path(output_path) if output_path is not None else PROCESSED_DIR / "city_day_features.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def train_model(df: pd.DataFrame) -> tuple[Pipeline, list[str], dict]:
    working = build_feature_frame(df)
    feature_columns = [c for c in working.columns if c not in {"AQI", "AQI_Bucket", "Date"}]
    categorical = [c for c in feature_columns if working[c].dtype == "object"]
    numeric = [c for c in feature_columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        (
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        working[feature_columns], working["AQI"], test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }
    return model, feature_columns, metrics


def save_model_artifact(model: Pipeline, feature_columns: Iterable[str]) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "random_forest_aqi.pkl"
    joblib.dump({"model": model, "features": list(feature_columns)}, target)
    return target


def save_metrics(metrics: dict, output_path: Path | str | None = None) -> Path:
    target = Path(output_path) if output_path is not None else REPORTS_DIR / "metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics, name="value").to_csv(target, header=True)
    return target


def run_full_pipeline() -> dict:
    raw_df = load_raw_dataset()
    cleaned = clean_dataframe(raw_df)
    clean_path = save_clean_data(cleaned)
    processed = build_feature_frame(cleaned)
    processed_path = save_processed_data(processed)
    model, feature_columns, metrics = train_model(cleaned)
    model_path = save_model_artifact(model, feature_columns)
    metrics_path = save_metrics(metrics)
    return {
        "clean_path": str(clean_path),
        "processed_path": str(processed_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "rows": len(cleaned),
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print(result)
