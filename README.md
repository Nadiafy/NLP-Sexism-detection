# NLP Assignment 1: Sexism Detection in Tweets

This repository contains the code, report, and results for a sexism detection task using both a BiLSTM model with GloVe embeddings and a fine-tuned RoBERTa transformer. The project was developed as part of the Master's Degree in Artificial Intelligence at the University of Bologna.

## Project Summary

The goal of this project is to classify tweets as *sexist* or *non-sexist* using the English subset of the **EXIST 2023 Task 1** dataset. Two models were implemented and evaluated:

* **BiLSTM + GloVe**: A recurrent model using pre-trained static word embeddings.
* **RoBERTa (cardiffnlp/twitter-roberta-base-hate)**: A transformer model fine-tuned for the task using domain-relevant pretraining.

### Evaluation Results

| Model   | Macro F1 (avg ± std) | Precision | Recall |
| ------- | -------------------- | --------- | ------ |
| BiLSTM  | 0.8302 ± 0.0066      | 0.842     | 0.819  |
| RoBERTa | 0.8680 ± 0.0036      | 0.881     | 0.855  |

The transformer model outperformed the BiLSTM baseline across all metrics.

---

## Repository Contents

* `Assignment1.ipynb` – Full pipeline including data preprocessing, model training, evaluation, and visualization.
* `Task 8 - Report.pdf` – Detailed academic report outlining methods, results, and error analysis.

---

## Dataset

* **Source**: [EXIST 2023 Task 1](https://exist2023.github.io)
* **Subset**: 6,000 English tweets labeled as sexist or non-sexist
* **Distribution**:

  * 60.4% Non-sexist
  * 39.6% Sexist

---

## Preprocessing Steps

* Text normalization (lowercasing)
* Removal of URLs, mentions, hashtags, and emojis
* Lemmatization using [SpaCy](https://spacy.io)
* Special handling for out-of-vocabulary tokens in GloVe

---

## Model Architectures

### BiLSTM

* Embeddings: `glove-wiki-gigaword-100` (100d, trainable)
* Architecture: BiLSTM with 64 units (bidirectional) followed by a dense sigmoid layer
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 32
* Epochs: 10
* Training Runs: 3 seeds \[42, 43, 44]

### RoBERTa

* Model: `cardiffnlp/twitter-roberta-base-hate` (pre-trained on tweets and hate speech)
* Fine-tuning Framework: Hugging Face Transformers
* Optimizer: AdamW
* Learning Rate: 1e-5
* Batch Size: 16
* Epochs: 5
* Class Weighting for Imbalance
* Early Stopping (patience=2, delta=0.01)
* Training Runs: 3 seeds \[42, 43, 44]

---

## Evaluation and Error Analysis

* **Metric Used**: Macro F1-score, averaged across seeds
* BiLSTM was prone to false negatives (missed subtle or implicit sexism) and false positives (misclassified gender-related but non-sexist statements).
* RoBERTa demonstrated better contextual understanding, though still struggled with sarcasm and highly implicit language.

---

## References

* Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* Pennington et al. (2014). [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [SpaCy](https://spacy.io)
* [EXIST 2023 Task](https://exist2023.github.io)

---

## Author

**Nadia Farokhpay**
Master’s Degree in Artificial Intelligence
University of Bologna
Email: [nadia.farokhpay@studio.unibo.it](mailto:nadia.farokhpay@studio.unibo.it)
