# 🧠 NLP Assignment 1: Sexism Detection in Tweets

This repository contains the code, report, and results for **sexism detection in tweets** using both a BiLSTM model with GloVe embeddings and a fine-tuned RoBERTa transformer. The project is part of the Master's in Artificial Intelligence program at the University of Bologna.

## 📄 Project Summary

We use the **EXIST 2023 Task 1** dataset to classify tweets as *sexist* or *non-sexist*. The study compares a traditional BiLSTM model using static word embeddings with a transformer-based model pre-trained on social media data:

- 📌 **BiLSTM + GloVe**: Serves as the baseline model.
- 📌 **RoBERTa (cardiffnlp/twitter-roberta-base-hate)**: Fine-tuned transformer for domain-specific classification.

### 📊 Key Results

| Model       | Macro F1 (avg ± std) | Precision | Recall |
|-------------|-----------------------|-----------|--------|
| BiLSTM      | 0.8302 ± 0.0066       | 0.842     | 0.819  |
| RoBERTa     | 0.8680 ± 0.0036       | 0.881     | 0.855  |

---

## 🛠️ Code Overview

- `Assignment1.ipynb`: Main notebook implementing preprocessing, BiLSTM, RoBERTa fine-tuning, and evaluation.
- `Report.pdf`: Formal academic report detailing the methods, results, and analysis.
- `summarized_report.pdf`: Shorter version of the report (optional, if you include it).

---

## 📂 Dataset

- **Source**: [EXIST 2023 Task 1](https://exist2023.github.io)
- **Subset used**: 6,000 English tweets
- **Class distribution**:
  - 60.4% Non-sexist
  - 39.6% Sexist

---

## 🔍 Preprocessing

- Lowercasing
- Removal of URLs, mentions, hashtags, emojis
- Lemmatization using [SpaCy](https://spacy.io)

---

## 🧪 Model Details

### BiLSTM
- Embeddings: `glove-wiki-gigaword-100` (81.85% coverage)
- Architecture: 2-layer BiLSTM + Dense Sigmoid
- Optimizer: Adam, LR = 0.001

### RoBERTa
- Model: `cardiffnlp/twitter-roberta-base-hate` (pre-trained on hate speech)
- Fine-tuning: Hugging Face Transformers
- Optimizer: AdamW, LR = 1e-5
- Early stopping on macro F1

---

## 🔎 Error Analysis

- **BiLSTM struggled** with implicit bias and gender-neutral phrases.
- **RoBERTa** reduced false positives and false negatives but struggled with:
  - Sarcasm and irony
  - Contextual references beyond tweet text

---

## 🧠 Future Work

- Data augmentation (e.g., back-translation)
- Sarcasm detection integration
- SHAP/LIME for interpretability
- Multi-modal signals (e.g., user metadata)

---

## 📚 References

- [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
- [GloVe (Pennington et al., 2014)](https://aclanthology.org/D14-1162)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SpaCy NLP Library](https://spacy.io)
- [EXIST 2023 Task](https://exist2023.github.io)

---

## 👩‍💻 Author

**Nadia Farokhpay**  
_M.Sc. in Artificial Intelligence_  
[Email](mailto:nadia.farokhpay@studio.unibo.it)
