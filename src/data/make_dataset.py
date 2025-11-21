import click
import logging
from pathlib import Path
import zipfile
import pandas as pd

from src.data.s3_utils import (
    get_s3_client,
    download_file_from_s3,
    upload_file_to_s3,
)


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option("--bucket", default="iris-datasets", help="S3 bucket name")
@click.option("--raw_key", default="iris.zip", help="S3 key of the raw dataset")
@click.option("--processed_key", default="iris_processed.csv", help="S3 key for processed file")
@click.option("--endpoint", default="http://localhost:9000", help="S3 endpoint URL")
@click.option("--access_key", default="minioadmin", help="S3 access key")
@click.option("--secret_key", default="minioadmin", help="S3 secret key")
def main(
    input_filepath: str,
    output_filepath: str,
    bucket: str,
    raw_key: str,
    processed_key: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Runs data processing scripts to turn raw data from S3 into cleaned data ready
    to be analyzed.
    """

    logger = logging.getLogger(__name__)
    logger.info("Starting data pipeline from S3")

    client = get_s3_client(endpoint, access_key, secret_key)
    download_file_from_s3(client, bucket, raw_key, Path(input_filepath))

    input_zip = Path(input_filepath)
    output_csv = Path(output_filepath)

    logger.info(f"Processing zip file: {input_zip}")

    with zipfile.ZipFile(input_zip, "r") as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("В zip нет CSV файла")
        zip_ref.extract(csv_files[0], input_zip.parent)
        csv_path = input_zip.parent / csv_files[0]

    df = pd.read_csv(csv_path)

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = (
        df[numeric_cols] - df[numeric_cols].mean()
    ) / df[numeric_cols].std()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info(f"Processed CSV saved to {output_csv}")
    upload_file_to_s3(client, bucket, output_csv, processed_key)
    logger.info(f"Uploaded processed file to S3 as {processed_key}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
