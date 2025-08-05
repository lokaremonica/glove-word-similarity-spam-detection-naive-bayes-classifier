# 🧠 GloVe Word Vector Analysis + Naive Bayes Spam Detection

This project covers two key NLP components:
1. Word similarity and analogy tasks using **GloVe embeddings**.
2. SMS spam classification using a **Naive Bayes Classifier** with handcrafted features.

---

## 📦 Part 1: GloVe Word Embedding Tasks

### 🔹 GloVe Download and Conversion

The notebook:

* Downloads the `glove.6B.300d.txt` embedding from Stanford
* Converts it to Word2Vec format using `gensim` for efficient loading
* Loads into a `KeyedVectors` model

```python
model = KeyedVectors.load_word2vec_format("glove.6B.300d.word2vec")
```

---

### 🔍 Word Similarity (Cosine)

```python
model.similarity("man", "woman") → 0.6999
model.similarity("chair", "throne") → 0.2755
```

---

### 🧠 Word Analogies

Examples:

* `"queen" is to "king" as "woman" is to "man"` → Result: `"queen"`
* `"girl" is to "woman" as "child" is to "adult"` → Result: `"mother"`

---

## 📦 Part 2: SMS Spam Classification (Naive Bayes)

* Dataset: [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* Total messages: 5,572
* Labels: `"spam"` or `"ham"`

### ✍️ Feature Engineering

* `$` count
* Digit length and counts
* Message length flags
* URL presence
* Emoticons (e.g. `:)`, `:(`, `:p`)
* Bigrams from cleaned + stemmed tokens

### 🔁 Pipeline

* Clean and stem messages
* Extract features into a NumPy array
* Use `DataFrameMapper` + `CountVectorizer` to create final input matrix
* Apply `MultinomialNB` from `sklearn`

---

## 🧪 Model Evaluation

* **Train/Test Split**
* Accuracy Score
* Confusion Matrix
* Cross-Validation

Visualized using `scikit-plot` and `matplotlib`.

---

## 📈 Sample Output

| Metric    | Value          |
| --------- | -------------- |
| Accuracy  | \~98%          |
| CV Score  | \~96%          |
| ROC-AUC   | Plotted curve  |
| Confusion | Plotted matrix |

---
