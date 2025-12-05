import boto3
from pathlib import Path
from typing import Any


def get_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
) -> Any:
    """Создаёт S3 клиента для работы с MinIO"""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def download_file_from_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    local_path: Path,
) -> None:
    """Скачивает файл из S3 в локальную папку"""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(local_path))
    print(f"Downloaded {key} from bucket {bucket} to {local_path}")


def upload_file_to_s3(
    s3_client: Any,
    bucket: str,
    local_path: Path,
    key: str,
) -> None:
    """Загружает локальный файл в S3"""
    s3_client.upload_file(str(local_path), bucket, key)
    print(f"Uploaded {local_path} to bucket {bucket} as {key}")


def upload_model_to_s3(
    s3_client: Any,
    bucket: str,
    local_model_path: Path,
    experiment_name: str,
    model_filename: str = "model.pkl",
) -> None:
    """Загружает модель в S3 в папку с названием эксперимента"""
    s3_key = f"{experiment_name}/{model_filename}"
    upload_file_to_s3(s3_client, bucket, local_model_path, s3_key)
