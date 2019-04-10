import argparse
import pandas as pd
from sklearn.metrics import classification_report
from ps3.common import Feature, read_data, vect_transform, load_model,vect_transform_tf
from sklearn.metrics import accuracy_score


def test():
    parser = argparse.ArgumentParser(description="Uses a trained model to identify a feature of sentences.")
    parser.add_argument("feature", metavar="FEATURE", type=Feature, choices=[f for f in Feature],
                        help="the feature to identify")
    parser.add_argument("test", metavar="TRAINING_DATA", type=str,
                        help="the path to the test corpra")
    parser.add_argument("model", metavar="MODEL_FILE", type=str,
                        help="the path to the model to load")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Print a classification report")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Read data from file into a data frame
    data = read_data(args.test)

    # For the 'issue' feature, discard data with an issue of 'NONE'
    if args.feature == Feature.issue:
        is_none = data['issue'] != 'NONE'
        data_keep_none = data[data['issue'] == 'NONE']
        data = data[is_none]


    # Classify sentences
    if (args.feature.value == 'genre'):
        x_test = vect_transform_tf(data)
    else:
        x_test = vect_transform(data)
    y_test = data[args.feature.value]

    # if ()
    predict = model.predict(x_test)


    # Prepare output
    output = data
    output[args.feature.value] = predict
    # print(data_keep_none)
    if args.feature == Feature.issue:
        output = pd.concat((output,data_keep_none))
    # Print each row in the (normalized) input format
    for _, r in output.iterrows():
        print(r['id'], r['sentence'], r['polarity'], r['issue'], r['genre'], sep="\t")

    if args.report:
        # Print report
        print()
        print(classification_report(y_test, predict))

        #print final accuracy
        print("The accuracy score is : ")
        print(accuracy_score(y_test, predict))