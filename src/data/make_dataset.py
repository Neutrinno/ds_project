# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()  # type: ignore
@click.argument('input_filepath', type=click.Path(exists=True))  # type: ignore
@click.argument('output_filepath', type=click.Path())  # type: ignore
def main(input_filepath: str, output_filepath: str) -> None:

    """Runs data processing scripts to turn raw data from (../raw) into
       cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
