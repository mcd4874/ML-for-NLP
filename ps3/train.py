import argparse
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from ps3.common import Feature, read_data, vect_transform, save_model,vect_transform_tf
from imblearn.over_sampling import RandomOverSampler

feature_models = {
    Feature.polarity: LogisticRegression(),
    Feature.genre: LinearSVC(),
    Feature.issue: LinearSVC()
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
    print(args.feature.value)
    if (args.feature.value == "genre"):
        print("got here")
        x_train = vect_transform_tf(data)

    else:
        x_train = vect_transform(data)

    print(x_train.shape)

    y_train = data[args.feature.value]

    if (args.feature.value == 'polarity'):
        ros = RandomOverSampler(random_state=42)
        x_train, y_train = ros.fit_resample(x_train, y_train)



    # Create and train the model
    if args.feature in feature_models:
        model = feature_models[args.feature]
    else:
        model= LinearSVC()
    model.fit(x_train, y_train)
    # Save the model to file for reuse
    save_model(model, args.out)
