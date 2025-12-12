import click
import itertools
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from src.data.s3_utils import get_s3_client
from src.models.train_model import load_config, train_model

logger = logging.getLogger(__name__)


def generate_hyperparameter_combinations(
    hyperparameters_grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Генерирует все комбинации гиперпараметров из grid"""
    keys = list(hyperparameters_grid.keys())
    values = list(hyperparameters_grid.values())

    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def run_experiments(
    config_path: Path,
    dataset_path: Path,
    mlflow_uri: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    dataset_s3_key: Optional[str] = None,
    use_docker: bool = False,
) -> None:
    """Запускает grid search эксперименты"""

    config = load_config(config_path)
    experiment_name = config.get("experiment_name", "default_experiment")
    hyperparameters_grid = config.get("hyperparameters_grid", {})

    if not hyperparameters_grid:
        raise ValueError("Config must contain 'hyperparameters_grid' for grid search")

    combinations = generate_hyperparameter_combinations(hyperparameters_grid)
    total_experiments = len(combinations)

    logger.info(f"Starting grid search: {total_experiments} experiments")
    logger.info(f"Hyperparameters grid: {hyperparameters_grid}")

    s3_client = get_s3_client(endpoint, access_key, secret_key)

    for i, hyperparams in enumerate(combinations, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"Experiment {i}/{total_experiments}")
        logger.info(f"Hyperparameters: {hyperparams}")
        logger.info("=" * 60)

        experiment_config = {
            "experiment_name": experiment_name,
            "model_type": config.get("model_type", "RandomForestClassifier"),
            "hyperparameters": hyperparams,
        }

        if use_docker:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp_config:
                yaml.dump(experiment_config, tmp_config)
                tmp_config_path = tmp_config.name

            try:
                project_name = Path.cwd().name.lower().replace("-", "_")
                network_name = f"{project_name}_default"

                check_network = subprocess.run(
                    ["docker", "network", "inspect", network_name],
                    capture_output=True,
                    text=True,
                    check=False
                )

                if check_network.returncode != 0:
                    inspect_mlflow = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}",
                            "mlflow"
                        ],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if inspect_mlflow.returncode == 0 and inspect_mlflow.stdout.strip():
                        network_id = inspect_mlflow.stdout.strip()
                        get_network_name = subprocess.run(
                            ["docker", "network", "inspect", "--format", "{{.Name}}", network_id],
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        if get_network_name.returncode == 0:
                            network_name = get_network_name.stdout.strip()
                            check_network.returncode = 0

                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{Path.cwd()}:/workspace",
                    "-w", "/workspace",
                ]

                if check_network.returncode == 0:
                    cmd.extend(["--network", network_name])
                    mlflow_uri_docker = mlflow_uri
                    endpoint_docker = endpoint
                else:
                    cmd.append("--network=host")
                    mlflow_uri_docker = mlflow_uri.replace("mlflow", "localhost")
                    endpoint_docker = endpoint.replace("minio", "localhost")

                cmd.extend([
                    "ds-train:latest",
                    "python", "-m", "src.models.train_model",
                    tmp_config_path,
                    str(dataset_path),
                    "--mlflow-uri", mlflow_uri_docker,
                    "--bucket", bucket,
                    "--endpoint", endpoint_docker,
                    "--access-key", access_key,
                    "--secret-key", secret_key,
                ])

                if dataset_s3_key is not None:
                    cmd.extend(["--dataset-s3-key", dataset_s3_key])

                subprocess.run(cmd, check=True)
            finally:
                os.unlink(tmp_config_path)

        else:
            train_model(
                dataset_path=dataset_path,
                config=experiment_config,
                experiment_name=experiment_name,
                mlflow_tracking_uri=mlflow_uri,
                s3_client=s3_client,
                s3_bucket=bucket,
                s3_endpoint=endpoint,
                s3_access_key=access_key,
                s3_secret_key=secret_key,
                dataset_s3_key=dataset_s3_key,
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"Grid search completed: {total_experiments} experiments finished")
    logger.info("=" * 60)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path())
@click.option("--mlflow-uri", default="http://localhost:5000", help="MLFlow tracking URI")
@click.option("--bucket", default="iris-datasets", help="S3 bucket name")
@click.option("--endpoint", default="http://localhost:9000", help="S3 endpoint URL")
@click.option("--access-key", default="minioadmin", help="S3 access key")
@click.option("--secret-key", default="minioadmin", help="S3 secret key")
@click.option(
    "--dataset-s3-key",
    default=None,
    help="S3 key for dataset (if downloading from S3)",
)
@click.option("--docker", is_flag=True, help="Run each experiment in Docker container")
def main(
    config_path: str,
    dataset_path: str,
    mlflow_uri: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    dataset_s3_key: Optional[str],
    docker: bool,
) -> None:
    """Запускает grid search эксперименты с различными комбинациями гиперпараметров"""

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    run_experiments(
        config_path=Path(config_path),
        dataset_path=Path(dataset_path),
        mlflow_uri=mlflow_uri,
        bucket=bucket,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        dataset_s3_key=dataset_s3_key,
        use_docker=docker,
    )


if __name__ == "__main__":
    main()
