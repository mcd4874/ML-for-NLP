import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from ps3.common import read_data, load_model


def main():
    parser = argparse.ArgumentParser(description="Trains a model to identify the source genre of sentences.")
    parser.add_argument("model", metavar="MODEL_FILE", type=str,
                        help="the path to the model to load")
    parser.add_argument("test", metavar="TRAINING_DATA", type=str,
                        help="the path to the test corpra")
    args = parser.parse_args()

    # Read in the test data
    data = read_data(args.test)

    # Load the model
    model = load_model(args.model)

    # Vectorize sentences
    cv = CountVectorizer().fit(data["sentence"])

    # Classify sentences
    x_test = cv.transform(data['sentence'])
    y_test = data['genre']
    predict = model.predict(x_test)

    # Print report
    print(classification_report(y_test, predict))


if __name__ == "__main__":
    main()
