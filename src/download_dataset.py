import os
from dataclasses import dataclass
from os import path

import click
import gdown


@dataclass
class GoogleDriveFile:
    id: str
    filename: str


gdrive_files = {
    "train": GoogleDriveFile(
        "1aAf5D2OeNuZ-1_8vEDy-N-n9sOuqAbjj",
        "train.pkl",
    ),
    "test": GoogleDriveFile(
        "1nwdtdoM0iA-6_KoE4CT9D1cWYyW6gMae",
        "test.pkl",
    ),
    "validation": GoogleDriveFile(
        "1Y5jAomCU0cW4bDAwRTBGmLGuAos1HxPG",
        "validation.pkl",
    ),
}


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
def main(key: str, output_folder: str) -> None:
    download_and_extract(key, output_folder)


if __name__ == "__main__":
    main()
