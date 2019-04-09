import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ps3.common import Feature, read_data, vect_transform, save_model

feature_models = {
    Feature.polarity: LogisticRegression()
}


def train():
    parser = argparse.ArgumentParser(description="Trains a model to identify a feature of sentences.")
    parser.add_argument("feature", metavar="FEATURE", type=Feature, choices=[f for f in Feature],
                        help="the feature for which to train")
    parser.add_argument("train", metavar="TRAINING_DATA", type=str,
                        help="the path to the training corpra")
    parser.add_argument("out", metavar="OUTPUT_FILE", type=str,
                        help="where to save the trained model")
    args = parser.parse_args()

    # Read data from file into a data frame
    data = read_data(args.train)

    # For the 'issue' feature, discard data with an issue of 'NONE'
    if args.feature == Feature.issue:
        is_none = data["issue"] != "NONE"
        data = data[is_none]

    # Prepare training data
    x_train = vect_transform(data)
    y_train = data[args.feature.value]

    # Create and train the model
    model = feature_models[args.feature] if args.feature in feature_models else RandomForestClassifier()
    model.fit(x_train, y_train)

    # Save the model to file for reuse
    save_model(model, args.out)
