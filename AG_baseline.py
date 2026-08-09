#!/usr/bin/python3

import time

import pandas as pd

from utils import load, prepare
from plots import plot_confusion_matrices

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


"""
How does the amount of training data affect the performance
of different text representations and classifiers,
particularly in low-resource settings?
"""


def evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    vectorizer,
    classifier,
):
    """
    Train and evaluate one NLP classification pipeline.

    The pipeline combines a text vectorizer with a classifier.
    Accuracy, Macro F1, training time, and inference time are measured.

    Returns
    -------
    dict
        Evaluation metrics and predictions for the pipeline.
    """

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])

    vectorizer_name = (vectorizer.__class__.__name__)
    classifier_name = (classifier.__class__.__name__)

    name = (f"{vectorizer_name} + {classifier_name}")

    # # # # # #
    # Training #
    # # # # # #

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = (time.perf_counter() - start)

    # # # # # # #
    # Prediction #
    # # # # # # #

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_time = (time.perf_counter() - start)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Training time : {train_time:.4f} s")
    print(f"Inference time: {inference_time:.4f} s")

    print(f"Accuracy      : {accuracy:.3f}")
    print(f"Macro F1      : {macro_f1:.3f}\n")

    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
        )
    )

    return {
        "Vectorizer": vectorizer_name,
        "Classifier": classifier_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Train time (s)": train_time,
        "Inference time (s)": inference_time,
        "y_true": y_test,
        "y_pred": predictions,
    }


def main():
    """
    Run baseline experiments on multiple training set sizes.

    Each experiment combines one text vectorizer with one
    classifier and evaluates the resulting pipeline on the
    fixed AG News test set.

    Training sizes:
        50, 200, 500, 2000, and 10000

    Test size:
        7600 observations
    """

    dataset_sizes = [50, 200, 500, 2000, 10000]

    experiments = [
            (CountVectorizer(), MultinomialNB()),
            (TfidfVectorizer(), MultinomialNB()),
            (CountVectorizer(), LinearSVC()),
            (TfidfVectorizer(), LinearSVC()),
    ]

    test = load("df_test.csv")

    if test is None:
        return

    X_test, y_test = prepare(test)
    print(f"Fixed test set: {len(X_test)} articles")

    results = []

    for size in dataset_sizes:

        print("\n")
        print("#" * 80)
        print(f"DATASET SIZE : {size}")
        print("#" * 80)

        train = load(f"ag_train_{size}.csv")

        if train is None:
            continue

        X_train, y_train = prepare(train)

        for vectorizer, classifier in experiments:

            result = evaluate(
                X_train,
                y_train,
                X_test,
                y_test,
                vectorizer,
                classifier,
            )

            result["Train size"] = size
            results.append(result)

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    display_columns = [
        "Train size",
        "Vectorizer",
        "Classifier",
        "Accuracy",
        "Macro F1",
        "Train time (s)",
        "Inference time (s)",
    ]

    print(results_df[display_columns].round({
            "Accuracy": 3,
            "Macro F1": 3,
            "Train time (s)": 4,
            "Inference time (s)": 4,
        })
    )

    plot_confusion_matrices(results_df)

    csv_columns = [
        "Train size",
        "Vectorizer",
        "Classifier",
        "Accuracy",
        "Macro F1",
        "Train time (s)",
        "Inference time (s)",
    ]

    results_df[csv_columns].to_csv(
        "results_baseline.csv", index=False,
    )


if __name__ == "__main__":
    main()
