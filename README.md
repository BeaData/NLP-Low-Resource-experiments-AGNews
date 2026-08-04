# AG News Text Classification: Classical NLP vs Sentence Embeddings

## 1. Project Overview

This project studies how different text representations and machine learning approaches perform for multi-class text classification under low-resource conditions.

The main research question is:

> How does training set size affect the performance of different text representations and classifiers on AG News classification?

The objective is not only to maximize accuracy, but to analyze the trade-offs between:

* representation quality,
* classifier complexity,
* training data requirements,
* computational cost.

The project compares classical NLP pipelines based on sparse lexical representations with modern sentence embedding approaches.

---

# 2. Dataset

The experiments use the **AG News dataset**, a benchmark dataset for news topic classification.

The dataset contains four categories:

| Label | Category |
| ----- | -------- |
| 1     | World    |
| 2     | Sports   |
| 3     | Business |
| 4     | Sci/Tech |

Each document is represented as:

```
Title + Description
```

## Experimental protocol

A fixed independent test set is used for evaluation.

Training subsets are nested:

```
50
200
500
2000
10000
```

This allows analysis of performance evolution when increasing the amount of labelled data.

---

# 3. Evaluation Metrics

The following metrics are reported:

## Accuracy

Overall proportion of correctly classified documents.

## Macro F1-score

Main evaluation metric.

Macro F1 gives equal importance to each class and is therefore more informative for multi-class comparison.

## Computational measurements

The experiments also record:

* text representation time,
* classifier training time,
* inference time.

---

# 4. Experimental Evolution

The project is organized into progressive experimental stages.

```
V1
 |
 |-- Baseline lexical representations
 |
V2
 |
 |-- N-gram representations
 |
V3
 |
 |-- Stemming
 |
V4
 |
 |-- Lemmatization
 |
V5
 |
 |-- Hyperparameter optimization
 |
V6
 |
 |-- Sentence embeddings
```

---

# 5. Classical NLP Experiments

## V1 - Baseline

Objective:

Establish a classical NLP reference.

Representations:

* CountVectorizer
* TF-IDF

Classifiers:

* Multinomial Naive Bayes
* LinearSVC

---

## V2 - N-grams

Objective:

Measure the contribution of local word order information.

Comparison:

* unigrams
* bigrams

---

## V3 - Stemming

Objective:

Evaluate morphological normalization.

Methods:

* Porter Stemmer
* Snowball Stemmer

---

## V4 - Lemmatization

Objective:

Compare linguistic normalization approaches.

Method:

* spaCy lemmatization

---

## V5 - Hyperparameter Optimization

Objective:

Determine whether optimized classical models can significantly improve over default configurations.

Method:

* GridSearchCV
* Stratified cross-validation
* Macro F1 optimization

Models optimized:

* MultinomialNB
* LinearSVC

---

# 6. Sentence Embedding Experiments

## V6 - Semantic Representations

Objective:

Evaluate whether dense semantic representations outperform classical sparse representations.

Embedding model:

```
all-MiniLM-L6-v2
```

Experiments:

---

## V6.1 Raw embeddings + LinearSVC

Pipeline:

```
Text
 |
Sentence Transformer
 |
Embedding vectors
 |
LinearSVC
```

---

## V6.2 L2 normalized embeddings + LinearSVC

Pipeline:

```
Text
 |
Sentence Transformer
 |
L2 normalization
 |
LinearSVC
```

Objective:

Measure whether embedding normalization improves classification.

---

## V6.3 Raw embeddings + Nearest Centroid

Pipeline:

```
Text
 |
Sentence Transformer
 |
Embedding vectors
 |
Class centroids
 |
Nearest centroid prediction
```

Objective:

Evaluate whether a simple classifier can exploit semantic embeddings efficiently, especially with limited training data.

---

# 7. Project Structure

Example:

```
AGNews/

├── data/
│
├── utils.py
├── plots.py
│
├── AG_V1_baseline.py
├── AG_V2_ngram.py
├── AG_V3_stemming.py
├── AG_V4_lemmatization.py
├── AG_V5_GridSearch.py
│
├── embedding_framework.py
│
├── results/
│
│   ├── results_V1.csv
│   ├── results_V2.csv
│   ├── results_V3.csv
│   ├── results_V4.csv
│   ├── results_V5.csv
│   ├── results_V6_1_embeddings.csv
│   ├── results_V6_2_embeddings.csv
│   └── results_V6_3_embeddings.csv
│
└── README.md
```

---

# 8. Environment

Recommended Python environment:

```
Python >= 3.12
```

Main dependencies:

```
numpy
pandas
scikit-learn
matplotlib
sentence-transformers
spacy
```

Installation:

```bash
uv sync
```

---

# 9. Running Experiments

Example:

```bash
python AG_V1_baseline.py
```

or:

```bash
python embedding_framework.py
```

Results are automatically saved as CSV files.

---

# 10. Main Questions Investigated

The project investigates:

1. How much performance is gained by improving lexical representations?
2. Do classical preprocessing techniques still matter?
3. Does hyperparameter optimization significantly improve classical models?
4. At what training size do semantic embeddings become advantageous?
5. Can very simple classifiers compete when using strong representations?

---

# 11. Expected Analysis

The final comparison will consider:

* Macro F1 ranking,
* Accuracy,
* training efficiency,
* inference cost,
* behavior under low-resource conditions.

---
## Acknowledgements

This project benefited from discussions and technical assistance provided by ChatGPT (OpenAI) and Deep Seek 3.2 Reasoner for code organization and documentation structure.
