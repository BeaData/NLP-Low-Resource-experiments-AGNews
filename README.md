# AG News Text Classification: From Classical NLP to Word and Sentence Embeddings

## 1. Project Overview

This project studies how different text representations and machine learning approaches perform for multi-class text classification under low-resource conditions.

The main research question is:

> How does training set size affect the performance of different text representations and classifiers on AG News classification?

The objective is not only to maximize accuracy, but to analyze the trade-offs between:

* representation quality,
* classifier complexity,
* training data requirements,
* computational cost.

The project progressively compares classical NLP pipelines based on sparse lexical representations with sentence-level and word-level embedding approaches.

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

```text
Title + Description
```

## Experimental protocol

A fixed independent test set is used for evaluation.

Training subsets are nested:

```text
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

```text
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
 |
V7
 |
 |-- Word embeddings + pooling
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

Evaluate whether dense sentence-level semantic representations outperform classical sparse representations.

Embedding model:

```text
all-MiniLM-L6-v2
```

---

## V6.1 - Raw Embeddings + LinearSVC

Pipeline:

```text
Text
 |
Sentence Transformer
 |
Embedding vectors
 |
LinearSVC
```

---

## V6.2 - L2-Normalized Embeddings + LinearSVC

Pipeline:

```text
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

## V6.3 - Raw Embeddings + Nearest Centroid

Pipeline:

```text
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

# 7. Word Embedding Experiments

## V7 - Word-Level Semantic Representations

Objective:

Evaluate whether word-level embeddings combined with pooling strategies can provide competitive document representations under low-resource conditions.

Embedding model:

```text
GloVe 100d
```

The word embeddings are aggregated at document level using different pooling strategies.

---

## V7.1 - Mean Pooling

Pipeline:

```text
Text
 |
Tokenization
 |
GloVe word embeddings
 |
Mean pooling
 |
LinearSVC
```

Objective:

Establish a simple word-level embedding baseline using the mean of the available word vectors.

---

## V7.2 - TF-IDF Weighted Mean Pooling

Pipeline:

```text
Text
 |
Tokenization
 |
GloVe word embeddings
 |
TF-IDF weighted pooling
 |
LinearSVC
```

Objective:

Evaluate whether weighting word embeddings according to their TF-IDF importance improves document representation compared with simple mean pooling.

---

# 8. Project Structure

Example:

```text
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
│   ├── results_V6_3_embeddings.csv
│   ├── results_V7_1_embeddings.csv
│   └── results_V7_2_embeddings.csv
│
└── README.md
```

---

# 9. Environment

Recommended Python environment:

```text
Python >= 3.12
```

Main dependencies:

```text
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

# 10. Running Experiments

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

# 11. Main Questions Investigated

The project investigates:

1. How much performance is gained by improving lexical representations?
2. Do classical preprocessing techniques still matter?
3. Does hyperparameter optimization significantly improve classical models?
4. At what training size do sentence-level semantic embeddings become advantageous?
5. Can simple classifiers compete when using strong representations?
6. Can word-level embeddings with simple pooling strategies compete with sentence-level embeddings?
7. How do representation quality and computational cost evolve across the different approaches?

---

## Acknowledgements

This project benefited from discussions and technical assistance provided by ChatGPT (OpenAI) and DeepSeek V3.2 Reasoner for code organization and documentation structure.
