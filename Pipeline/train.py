"""
train.py — XGBoost training script voor de Docker-container.

Spark wordt gebruikt voor data preprocessing. XGBoost doet de modeltraining.

Gebruik:
    docker run --rm \
        -v $(pwd):/app/output \
        -v $(pwd)/Data:/app/Data \
        -v $(pwd)/mlruns-cloud:/app/mlruns-cloud \
        traffic-spark python3 train.py --data Data/Metro_Interstate_Traffic_Volume.csv --output /app/output
"""

import argparse
import datetime
import json
import os
import sys

# PySpark locatie in de spark:python3 container
sys.path.insert(0, "/opt/spark/python")
sys.path.insert(0, "/opt/spark/python/lib/py4j-src.zip")

import mlflow
import mlflow.xgboost
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------
MLRUNS_DIR = os.getenv("MLRUNS_DIR", "mlruns-cloud")
MLFLOW_URI = f"sqlite:///{MLRUNS_DIR}/mlflow.db"
EXPERIMENT = "nova_cloud"
MODEL_NAME = "XGBTrafficModel"

PARAMS = dict(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.8,
    reg_alpha=0.4,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_spark():
    """Maak een SparkSession aan die werkt op Linux (in de container)."""
    return (
        SparkSession.builder.appName("NovaStadTrafficForecast")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def preprocess(df):
    """Feature-engineering: tijdsfeatures + one-hot encoding weer-kolommen."""
    df = (
        df.withColumn("Hour", F.hour("date_time"))
        .withColumn("DayOfWeek", F.dayofweek("date_time"))
        .withColumn("Month", F.month("date_time"))
        .withColumn("Year", F.year("date_time"))
        .withColumn(
            "holiday",
            F.when(F.col("holiday") == "None", False).otherwise(True),
        )
    )

    for val in [r[0] for r in df.select("weather_main").distinct().collect()]:
        df = df.withColumn(
            f"weather_main_{val}",
            F.when(F.col("weather_main") == val, True).otherwise(False),
        )

    for val in [r[0] for r in df.select("weather_description").distinct().collect()]:
        df = df.withColumn(
            f"weather_description_{val}",
            F.when(F.col("weather_description") == val, True).otherwise(False),
        )

    return df.drop("weather_main", "weather_description", "date_time")


def train(data_path, output_dir):
    """Laad data via Spark, train XGBoost, log naar MLflow."""
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT)

    # Data laden en preprocessen via Spark
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = preprocess(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count()}  Test: {test_df.count()}")

    # Spark → Pandas voor XGBoost
    feature_cols = [c for c in df.columns if c != "traffic_volume"]
    train_pd = train_df.toPandas()
    test_pd  = test_df.toPandas()

    X_train = train_pd[feature_cols].astype(float)
    y_train = train_pd["traffic_volume"].astype(float)
    X_test  = test_pd[feature_cols].astype(float)
    y_test  = test_pd["traffic_volume"].astype(float)

    with mlflow.start_run(run_name="xgb_traffic") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("timestamp", datetime.datetime.now().isoformat())

        model = XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae  = float(mean_absolute_error(y_test, preds))
        r2   = float(r2_score(y_test, preds))
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        baseline_rmse = float(np.sqrt(
            mean_squared_error(y_test, [float(y_train.mean())] * len(y_test))
        ))
        mlflow.log_metric("baseline_rmse", baseline_rmse)

        mlflow.xgboost.log_model(
            model, name=MODEL_NAME, registered_model_name=MODEL_NAME
        )
        run_id = run.info.run_id

    metrics = {
        "run_id": run_id,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "baseline_rmse": baseline_rmse,
        "improvement_pct": (1 - rmse / baseline_rmse) * 100,
    }

    # Schrijf metrics.json naar de gemounte output-map op de host
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"metrics.json geschreven naar {output_path}")
    spark.stop()
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="Data/Metro_Interstate_Traffic_Volume.csv",
        help="Pad naar de CSV-dataset",
    )
    parser.add_argument(
        "--output",
        default="/app",
        help="Map waar metrics.json naartoe wordt geschreven",
    )
    args = parser.parse_args()
    train(args.data, args.output)