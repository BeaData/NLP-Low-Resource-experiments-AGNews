#!/usr/bin/python3

from utils import load, prepare
import pandas as pd


def create_nested_datasets(df: pd.DataFrame,  # type: ignore
                           output_prefix: str,
                           sizes: list[int]) -> None:
    """
    Create nested balanced datasets.

    Example:
        50 ⊂ 200 ⊂ 500 ⊂ 2000 ⊂ 10000
    """

    class_column = "Class Index"

    groups = {}

    # Shuffle each class once
    for label in sorted(df[class_column].unique()):
        groups[label] = (
            df[df[class_column] == label]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    n_classes = len(groups)

    for size in sizes:

        base = size // n_classes
        remainder = size % n_classes

        parts = []

        for i, label in enumerate(sorted(groups)):
            n = base + (1 if i < remainder else 0)
            parts.append(groups[label].iloc[:n])

        subset = pd.concat(parts, ignore_index=True)

        subset = subset.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        filename = f"{output_prefix}_{size}.csv"
        subset.to_csv(filename, index=False)

        print(f"{filename:<20} {len(subset)} articles")


def main():
    """
    Create balanced nested training datasets and save
    the complete AG News test dataset.

    Training sizes:
        50, 200, 500, 2000, and 10000

    Test size:
        7600 observations
    """

    train = load("ag_train.csv")

    if train is None:
        return

    sizes = [50, 200, 500, 2000, 10000]

    print("Creating nested training datasets")
    create_nested_datasets(train, "ag_train", sizes)

    print()

    test = load("test.csv")

    if test is None:
        return

    text, labels = prepare(test)

    df_test = pd.DataFrame(
        {
            "Text": text,
            "Class Index": labels
        }
    )

    df_test.to_csv("df_test.csv", index=False)

    print(
        f"{'df_test.csv':<20} "
        f"{len(df_test)} articles"
    )


if __name__ == "__main__":
    main()
