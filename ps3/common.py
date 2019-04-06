import os
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords

PUNCT_TABLE = dict((ord(char), None) for char in string.punctuation.join(['“', '”', '’']))
STOPLIST = stopwords.words('english')


def clean_sentence(text):
    # Lowercase text
    text = text.lower()

    # Strip punctuation
    text = text.translate(PUNCT_TABLE)

    # Filter out English stop words
    text = " ".join([word for word in text.split() if word not in STOPLIST])

    return text


def read_data(file):
    # Read in training data
    data = pd.read_csv(os.path.abspath(file), delimiter="\t",
                       names=["index", "sentence", "polarity", "issue", "genre", "uname"])

    # Discard unnecessary columns
    data = data[["sentence", "polarity", "issue", "genre"]]

    # Clean sentences
    data["sentence"] = data["sentence"].apply(clean_sentence)

    return data


def save_model(model, file):
    with open(os.path.abspath(file), "wb") as model_out:
        pickle.dump(model, model_out)
        print(f"==> Model saved to: {file}")


def load_model(file):
    with open(os.path.abspath(file), "rb") as model_in:
        return pickle.load(model_in)
