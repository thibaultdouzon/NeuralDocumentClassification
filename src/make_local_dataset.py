import json
import pickle
from os import path
from typing import Any, Final

import click
import datasets
import pydantic
import tqdm
from typing_extensions import TypedDict

class_to_idx: Final = {
    "letter": 0,
    "form": 1,
    "email": 2,
    "handwritten": 3,
    "advertisement": 4,
    "scientific report": 5,
    "scientific publication": 6,
    "specification": 7,
    "file folder": 8,
    "news article": 9,
    "budget": 10,
    "invoice": 11,
    "presentation": 12,
    "questionnaire": 13,
    "resume": 14,
    "memo": 15,
}

idx_to_class: Final = list(class_to_idx.keys())


class DatasetSizes(TypedDict):
    train: int
    test: int
    validation: int


class DatasetDescription(pydantic.BaseModel):
    name: str
    sizes: DatasetSizes
    classes: list[str]


def load_dataset_descriptions(file: str) -> DatasetDescription:
    with open(file, "r") as f:
        dataset_descriptions = json.load(f)
    return DatasetDescription(**dataset_descriptions)


def prune_dataset(
    dataset: datasets.Dataset, dataset_description: DatasetDescription
) -> dict[str, list[dict[str, Any]]]:
    dataset = dataset.shuffle()

    new_dataset_data: dict[str, list[dict[str, Any]]] = {}
    for split in DatasetSizes.__annotations__.keys():  # ["train", "test", "validation"]
        new_dataset_data[split] = []
        with tqdm.tqdm(total=dataset_description.sizes[split], desc=split) as pbar:  # type: ignore
            for data in dataset[split]:
                if idx_to_class[data["label"]] in dataset_description.classes:
                    new_dataset_data[split].append(data)
                    pbar.update(1)
                if len(new_dataset_data[split]) == dataset_description.sizes[split]:  # type: ignore
                    break

    return new_dataset_data


@click.command()
@click.option(
    "--dataset_description_file",
    default="dataset_descriptions.json",
    help="Path to the dataset description file",
)
@click.option("--output_dir", default="dataset", help="Path to the output directory")
def main(dataset_description_file: str, output_dir: str) -> None:
    dataset_description = load_dataset_descriptions(dataset_description_file)

    full_dataset = datasets.load_dataset(dataset_description.name)

    dataset_splits = prune_dataset(full_dataset, dataset_description)

    for split, dataset in dataset_splits.items():
        with open(path.join(output_dir, f"{split}.pkl"), "wb") as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
