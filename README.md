# Toxic Comment Classification

The goal of this project is to build a model able to filter user comments based on the degree of harmfulness of the language. The main points to be addressed are the following:
<br>

**1. Preprocess the text by removing the set of tokens that do not provide significant semantic contribution**<br><br>

**2. Transform the text corpus into sequences**<br><br>

**3. Build a Deep Learning model including recurrent layers for a multilabel classification task**<br><br>

The project contains 4 different models:
- **Model 1**: LSTM
- **Model 2**: LSTM with weighted loss function
- **Model 3**: LSTM with CNN layers and weighted loss function
- **Model 4**: LSTM with CNN layers and rebalance of the dataset

There is also a model based on RNN and GRU layers but it is not included in this notebook. The model is available in the file `models.py`.<br><br>

The dataset contains the following columns:
- **id**: the unique id of the comment
- **comment_text**: the text of the comment
- **toxic**: 1 if the comment is toxic, 0 if not
- **severe_toxic**: 1 if the comment is severely toxic, 0 if not
- **obscene**: 1 if the comment is obscene, 0 if not
- **threat**: 1 if the comment contains a threat, 0 if not
- **insult**: 1 if the comment is insulting, 0 if not
- **identity_hate**: 1 if the comment contains hate speech, 0 if not

## Requirements

Install the required libraries by running the following command:
```bash
pip install -r requirements.txt
```