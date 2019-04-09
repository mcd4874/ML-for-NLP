import argparse
from sklearn.metrics import classification_report
from ps3.common import Feature, read_data, vect_transform, load_model


def test():
    parser = argparse.ArgumentParser(description="Uses a trained model to identify a feature of sentences.")
    parser.add_argument("feature", metavar="FEATURE", type=Feature, choices=[f for f in Feature],
                        help="the feature to identify")
    parser.add_argument("test", metavar="TRAINING_DATA", type=str,
                        help="the path to the test corpra")
    parser.add_argument("model", metavar="MODEL_FILE", type=str,
                        help="the path to the model to load")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Read data from file into a data frame
    data = read_data(args.test)

    # Classify sentences
    x_test = vect_transform(data)
    y_test = data[args.feature.value]
    predict = model.predict(x_test)

    # Print report
    print(classification_report(y_test, predict))
