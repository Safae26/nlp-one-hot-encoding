# NLP Preprocessing & One-Hot Encoding Hands On

Hello there, fellow AI enthusiast\! 👋 This Jupyter Notebook is a hands-on exercise exploring the initial stages of **Natural Language Processing (NLP)**, specifically focusing on **text preprocessing** and the classic method of **One-Hot Vector Representation** for words.

The goal is to take a raw text corpus and transform it into a numerical format that machine learning models can understand, step-by-step.

-----

## 🛠️ Dependencies

This notebook relies on the following popular Python libraries for NLP tasks:

| Library | Purpose |
| :--- | :--- |
| **`nltk`** | Tokenization and Stopword removal. |
| **`spacy`** | Advanced text processing, specifically for Lemmatization. |

-----

## Preprocessing Pipeline

The notebook walks through several crucial steps to clean and standardize the text before vectorization:

### 1\. Tokenization ✂️

The corpus is first broken down into **sentences** and then each sentence is broken down into individual **words**.

### 2\. Lowercasing ⬇️

All words are converted to **lowercase** to ensure that variations in capitalization (e.g., "AI", "Ai", "ai") are treated as the same word.

### 3\. Deleting Stopwords 🚫

Common, low-value words like "is," "a," "the," "and" (known as **stopwords**) are removed as they don't typically contribute much to the meaning or classification task.

### 4\. Lemmatization

Words are reduced to their base or dictionary form (the **lemma**). For example, if we had "thinking," it would be reduced to "think" (though in the notebook's run, "thinking" remains the lemma). Note how "data" is sometimes lemmatized to "**datum**" by the spaCy model.

-----

## 🔢 One-Hot Vectorization

After preprocessing, the cleaned text is ready for numerical representation.

### 1\. Vocabulary Construction

A unique set of all remaining words (the **vocabulary**) is built across the entire corpus.

  * **Vocabulary Size:** **20** unique terms.
  * **Example Terms:** `'explore'`, `'science'`, `'ai'`, `'datum'`, etc.

### 2\. Vocabulary Indexation

Each unique word in the vocabulary is assigned a unique integer index. This is the foundation for creating the vector.

### 3\. The One-Hot Vector Function

The `create_one_hot` function generates the vector for any given word:

  * The vector size equals the **Vocabulary Size (20)**.
  * The vector contains a **1** at the index corresponding to the word, and **0s** everywhere else.

For example, the word **`hello`** has an index of **9**. Its vector will be:
$$[0, 0, 0, 0, 0, 0, 0, 0, 0, \mathbf{1}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$$

*(Notice the '1' at the 10th position, corresponding to index 9.)*

The output `vectors_one_hot[0]` shows the vectors for the words in the first sentence (`['hello', '!']`):

```python
# Output for the first sentence's lemmatized tokens: ['hello', '!']
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```
