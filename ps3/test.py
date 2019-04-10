import argparse
import pandas as pd
from sklearn.metrics import classification_report
from ps3.common import Feature, read_data, load_model,transform
from sklearn.metrics import accuracy_score


def test():
    parser = argparse.ArgumentParser(description="Uses a trained model to identify a feature of sentences.")
    parser.add_argument("feature", metavar="FEATURE", type=Feature, choices=[f for f in Feature],
                        help="the feature to identify")
    parser.add_argument("test", metavar="TRAINING_DATA", type=str,
                        help="the path to the test corpra")
    parser.add_argument("model", metavar="MODEL_FILE", type=str,
                        help="the path to the model to load")
    parser.add_argument("text_model", metavar="TEXT_MODEL_FILE", type=str,
                        help="where to load the trained text model")
    parser.add_argument("report_file", metavar="REPORT_FILE", type=str,
                        help="generate report file")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Print a classification report")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Read data from file into a data frame
    data = read_data(args.test)



    # For the "issue" feature, discard data with an issue of "NONE"
    if args.feature == Feature.issue:
        is_none = data["issue"] != "NONE"
        data_keep_none = data[data["issue"] == "NONE"]
        data = data[is_none]

    out = data.copy()
    cv_model = load_model(args.text_model)

    # Classify sentences
    x_test = transform(data,cv_model)
    y_test = data[args.feature.value]

    predict = model.predict(x_test)



    # Prepare output
    out[args.feature.value] = predict
    if args.feature == Feature.issue:
        out = pd.concat((out, data_keep_none))
    out = out.sort_index()
    out.to_csv(args.report_file, index = False)

    # Print each row in the (normalized) input format

    if args.report:
        # Print final accuracy
        print("Overall accuracy score:", accuracy_score(y_test, predict))

        # Print report
        print()
        print(classification_report(y_test, predict))



    for _, r in out.iterrows():
        print(r["id"], r["sentence"], r["polarity"], r["issue"], r["genre"], sep="\t")


