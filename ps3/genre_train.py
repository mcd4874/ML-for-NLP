import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from ps3.common import read_data, save_model


def main():
    parser = argparse.ArgumentParser(description="Trains a model to identify the source genre of sentences.")
    parser.add_argument("train", metavar="TRAINING_DATA", type=str,
                        help="the path to the training corpra")
    parser.add_argument("--out", "-o", metavar="OUTPUT_FILE", type=str,
                        help="where to save the trained model")
    args = parser.parse_args()

    data = read_data(args.train)

    # Vectorize sentences
    cv = CountVectorizer().fit(data["sentence"])

    # Split and prepare training data
    x_train = cv.transform(data['sentence'])
    y_train = data['genre']

    # Create and train the classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(x_train, y_train)

    if args.out:
        # Save the model to file for reuse
        save_model(model, args.out)


if __name__ == "__main__":
    main()
