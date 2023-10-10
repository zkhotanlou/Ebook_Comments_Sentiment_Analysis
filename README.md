# Sentiment Analysis on eBook Comments

This repository contains the code and resources for a sentiment analysis project that analyzes customer comments about eBooks on an application. The goal of this project is to predict whether customers have a positive or negative feedback about each eBook based on their comments.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Loss and Accuracy Diagram](#loss-and-accuracy-diagram)
- [Prediction](#prediction)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)

  
## Introduction

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment or emotion expressed in a piece of text. In this project, we use the ParsBERT model, a transformer-based architecture, to perform sentiment analysis on customer comments about eBooks. The goal is to automatically classify comments as positive or negative based on their content.

## Dataset

We use a dataset of customer comments about eBooks collected from an application. The dataset includes comments and corresponding ratings provided by customers. To prepare the dataset for sentiment analysis, we perform data preprocessing, including text cleaning and label mapping. Ratings lower than 3.0 are labeled as negative, and ratings higher or equal to 3.0 are labeled as positive.

## Installation

To run this project, you'll need the following Python libraries:

- NumPy
- Pandas
- Scikit-learn
- Transformers (Hugging Face Transformers)
- Plotly
- TQDM
- Hazm

You can install these libraries using pip:

```bash
pip install numpy pandas scikit-learn transformers plotly tqdm hazm
```

## Usage

To train the sentiment analysis model using the ParsBERT transformer, follow these steps:

1. Clone the repository:
2. Run the Jupyter Notebook for model training. The notebook contains the complete pipeline for data preprocessing, model training, and evaluation.

## Preprocessing

1. Cleaning commets including:
   - fixing unicodes
   - removing specials like a phone number, email, url, new lines, ...
   - cleaning HTMLs
   - normalizing
   - removing emojis
   - removing extra spaces, hashtags
2. Remove Comments with the Length of Fewer than 3 Words & More than 256 Words
3. Handling Unbalanced Data

## Model

1. Setup the Tokenizer and Configuration
2. Input Embeddings
3. Define Adam Optimizer, Scheduler & Cross Entropy Loss Function

ParsBERT is a monolingual model based on the BERT architecture from Google, but it has been trained on a large volume of Persian texts on various topics. This model has much higher accuracy for Persian language processing applications, including sentiment analysis and text classification, compared to the multilingual BERT model [^1].

## Loss and Accuracy Diagram

![image](https://github.com/zkhotanlou/Ebook_Comments_Sentiment_Analysis/assets/84021970/5c1fef5d-1126-4d49-9792-60354156940a)

![image](https://github.com/zkhotanlou/Ebook_Comments_Sentiment_Analysis/assets/84021970/9af9f577-70fe-4f25-8cd2-5804fa1be489)

## Prediction

After training the model, you can use it to make predictions on new comments. Here's an example of how to use the prediction function:
```python
test_comments = test['comment'].to_numpy()
preds, probs = predict(pt_model, test_comments, tokenizer, max_len=128)
```

## Results

The project results include:
- A trained sentiment analysis model based on ParsBERT.
- Evaluation metrics such as F1-score, precision, and recall.
- Predictions and probabilities for new comments.

## References

1. [ParsBERT: Transformer-based Model for Persian Language Understanding]([https://example.com/parsbert-paper](https://doi.org/10.1007%2Fs11063-021-10528-4))
   - Authors: Mehrdad Farahani and Mohammad Gharachorloo and Marzieh Farahani and Mohammad Manthouri
   - Published: October 2021
   - Journal: Neural Processing Letters

### Contributing

Contributions are welcome! If you have ideas for improving the project or want to collaborate, please open an issue or create a pull request.

