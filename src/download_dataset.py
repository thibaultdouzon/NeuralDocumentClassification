import os
from dataclasses import dataclass
from os import path
from typing import Optional

import click
import gdown
import requests
import tqdm


@dataclass
class GoogleDriveFile:
    id: str
    filename: str


gdrive_files = {
    "train": GoogleDriveFile(
        "1kMgms2dWKbQdzcsJ9YnwXPZrzeTy0z1w",
        "train.pkl",
    ),
    "test": GoogleDriveFile(
        "1zXMLtOeVq_QX6V5PB_gqAg9SMYpzC4eX",
        "test.pkl",
    ),
    "validation": GoogleDriveFile(
        "1gusWsXJ3BEpviWBMLzU8pWzvHEUWT4Wd",
        "validation.pkl",
    ),
}


@click.command()
@click.option(
    "--key",
    help="Key to download from Google Drive [train | test | validation | all]",
)
@click.option(
    "--output_folder",
    default="./dataset/",
    help="Path to save the downloaded file",
)
def download_and_extract(key: str, output_folder: str) -> None:
    assert key in ["train", "test", "validation", "all"]
    if key == "all":
        keys = ["train", "test", "validation"]
    else:
        keys = [key]

    for key in keys:
        remote_file = gdrive_files[key]

        if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        destination_file = path.join(output_folder, remote_file.filename)
        if not os.path.exists(destination_file):
            print(f"Downloading {destination_file} from Google Drive")
            gdown.download(id=remote_file.id, output=destination_file, quiet=False)


if __name__ == "__main__":
    download_and_extract()
