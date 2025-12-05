import click
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import yaml  # type: ignore[import-untyped]

from src.data.s3_utils import (
    get_s3_client,
    download_file_from_s3,
    upload_model_to_s3,
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Загружает конфиг из YAML файла"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), f"Config file {config_path} must contain a dictionary"
    return config  # type: ignore[return-value]


def train_model(
    dataset_path: Path,
    config: Dict[str, Any],
    experiment_name: str,
    mlflow_tracking_uri: str,
    s3_client: Any,
    s3_bucket: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    dataset_s3_key: Optional[str] = None,
) -> None:
    """Обучает модель с заданными гиперпараметрами и логирует в MLFlow"""

    # Если указан S3 ключ, скачиваем датасет из S3
    if dataset_s3_key:
        logger.info(f"Downloading dataset from S3: {dataset_s3_key}")
        download_file_from_s3(s3_client, s3_bucket, dataset_s3_key, dataset_path)

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Удаляем Id колонку если есть (утечка данных)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        logger.info("Dropped 'Id' column to prevent data leakage")

    # Последняя колонка - целевая переменная
    # Для Iris: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Dataset shape: {X.shape}, target: {target_col}")
    logger.info(f"Features: {list(feature_cols)}")

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Настройка MLFlow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Гиперпараметры из конфига
    hyperparams = config["hyperparameters"]
    model_type = config.get("model_type", "RandomForestClassifier")

    logger.info(f"Training {model_type} with hyperparameters: {hyperparams}")

    with mlflow.start_run():
        # Создание модели
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", None),
                max_features=hyperparams.get("max_features", "sqrt"),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Обучение
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Вычисление метрик
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }

        logger.info(f"Metrics: {metrics}")

        # Логирование в MLFlow
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics)

        # Сохранение модели локально
        model_dir = Path("models") / experiment_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")

        # Логирование модели в MLFlow (с обработкой ошибок)
        try:
            mlflow.sklearn.log_model(model, "model")
            logger.info("Model logged to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log model to MLFlow: {e}")
            logger.warning("Continuing without MLFlow model artifact")

        # Загрузка модели в S3
        upload_model_to_s3(
            s3_client,
            s3_bucket,
            model_path,
            experiment_name,
            "model.pkl",
        )

        logger.info(f"Model uploaded to S3: {experiment_name}/model.pkl")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path())
@click.option("--mlflow-uri", default="http://localhost:5000", help="MLFlow tracking URI")
@click.option("--bucket", default="iris-datasets", help="S3 bucket name")
@click.option("--endpoint", default="http://localhost:9000", help="S3 endpoint URL")
@click.option("--access-key", default="minioadmin", help="S3 access key")
@click.option("--secret-key", default="minioadmin", help="S3 secret key")
@click.option("--dataset-s3-key", default=None, help="S3 key for dataset (if downloading from S3)")
def main(
    config_path: str,
    dataset_path: str,
    mlflow_uri: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    dataset_s3_key: Optional[str],
) -> None:
    """Обучает модель согласно конфигу и логирует результаты в MLFlow"""

    logger.info("Starting model training")

    config = load_config(Path(config_path))
    experiment_name = config.get("experiment_name", "default_experiment")

    s3_client = get_s3_client(endpoint, access_key, secret_key)

    dataset_path_obj = Path(dataset_path)
    # Если файл не существует локально и указан S3 ключ, создаем путь для скачивания
    if not dataset_path_obj.exists() and dataset_s3_key:
        dataset_path_obj.parent.mkdir(parents=True, exist_ok=True)

    train_model(
        dataset_path=dataset_path_obj,
        config=config,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_uri,
        s3_client=s3_client,
        s3_bucket=bucket,
        s3_endpoint=endpoint,
        s3_access_key=access_key,
        s3_secret_key=secret_key,
        dataset_s3_key=dataset_s3_key,
    )

    logger.info("Training completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
