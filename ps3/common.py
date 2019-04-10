import os
import re
import pickle
import enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Feature(enum.Enum):
    genre = 'genre'
    polarity = 'polarity'
    issue = 'issue'

    def __str__(self):
        return self.value


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # Convert text to lowercase
    text = text.strip().lower()

    # Replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


def read_data(file):
    # Read in training data
    data = pd.read_csv(os.path.abspath(file), delimiter=r"\t|\s{3,}",
                       names=["index", "sentence", "polarity", "issue", "genre", "uname"])

    # Discard unnecessary columns
    data = data[["sentence", "polarity", "issue", "genre"]]
    data["issue"] = data["issue"].str.strip()
    return data


def vect_transform(data):
    cv = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text
    ).fit(data["sentence"])

    return cv.transform(data['sentence'])

def vect_transform_tf(data):
    cv = TfidfVectorizer(stop_words="english",preprocessor=clean_text,ngram_range=(1, 2)).fit(data["sentence"])

    return cv.transform(data['sentence'])


def save_model(model, file):
    with open(os.path.abspath(file), "wb") as model_out:
        pickle.dump(model, model_out)
        print(f"Model saved to: {file}")


def load_model(file):
    with open(os.path.abspath(file), "rb") as model_in:
        return pickle.load(model_in)
